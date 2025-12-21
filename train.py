import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path

import config
import models
import losses
import data_utils
import utils

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE_ID
    
    segments = data_utils.load_and_preprocess_data()
    train_loader, val_loader, test_loader, feature_scaler, target_scaler, test_segs = data_utils.get_data_loaders(segments)
    
    target_min_vals = torch.tensor(target_scaler.min_, dtype=torch.float32, device=config.device)
    target_scale_vals = torch.tensor(target_scaler.data_max_ - target_scaler.data_min_, dtype=torch.float32, device=config.device)
    
    model = models.PhysicsBiLSTMAttn(
        input_dim=len(config.FEATURE_COLUMNS),
        hidden_dim=config.HIDDEN_DIM,
        out_steps=config.TIME_STEPS_OUT,
        n_targets=len(config.TARGET_COLUMNS),
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        num_heads=config.NUM_HEADS
    ).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    target_col_map = {col_name: idx for idx, col_name in enumerate(config.TARGET_COLUMNS)}
    
    physics_loss_fn = losses.PhysicsInformedLoss(
        target_min_vals=target_min_vals,
        target_scale_vals=target_scale_vals,
        target_col_indices={
            'V_terminal': target_col_map["Cell_voltage"],
            'I_future': target_col_map["total_current"],
            'Voc': target_col_map["Estimated_Voc_Kalman"],
            'Vp': target_col_map["Estimated_Vp_Kalman"],
            'Vo': target_col_map["Estimated_Vo_Kalman"],
            'Ro': target_col_map["Estimated_Ro_k_Kalman"]
        },
        lambda_physics=config.LAMBDA_PHYSICS,
        lambda_aux=config.LAMBDA_AUX,
        lambda_main=config.LAMBDA_MAIN
    ).to(config.device)

    best_val_loss = np.inf
    pat_cnt = 0
    save_path = config.OUTPUT_DIR / "best_model_physics_v2.pth"

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss_sum = 0
        train_count = 0
        
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Training", disable=True):
            xb, yb = xb.to(config.device), yb.to(config.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_scaled = model(xb)
                total_loss, _, _, _ = physics_loss_fn(pred_scaled, yb)

                V_terminal_pred_s = pred_scaled[:, :, target_col_map["Cell_voltage"]]
                V_terminal_true_s = yb[:, :, target_col_map["Cell_voltage"]]
                loss_deriv_V_term = losses.derivative_loss_fn(V_terminal_pred_s, V_terminal_true_s) + losses.second_derivative_loss_fn(V_terminal_pred_s, V_terminal_true_s)

                Voc_pred_s = pred_scaled[:, :, target_col_map["Estimated_Voc_Kalman"]]
                Voc_true_s = yb[:, :, target_col_map["Estimated_Voc_Kalman"]]
                loss_deriv_Voc = losses.derivative_loss_fn(Voc_pred_s, Voc_true_s) + losses.second_derivative_loss_fn(Voc_pred_s, Voc_true_s)

                loss_deriv = loss_deriv_V_term + 0.3 * loss_deriv_Voc
                final_loss = total_loss + config.WEIGHT_DERIVATIVE * loss_deriv

            scaler.scale(final_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += final_loss.item()
            train_count += 1
        
        avg_train_loss = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0
        val_count = 0
        with torch.no_grad():
            for xb_val, yb_val in val_loader:
                xb_val, yb_val = xb_val.to(config.device), yb_val.to(config.device)
                pred_val_scaled = model(xb_val)
                val_total_loss, _, _, _ = physics_loss_fn(pred_val_scaled, yb_val)

                V_term_pred_val_s = pred_val_scaled[:, :, target_col_map["Cell_voltage"]]
                V_term_true_val_s = yb_val[:, :, target_col_map["Cell_voltage"]]
                val_loss_deriv_V_term = losses.derivative_loss_fn(V_term_pred_val_s, V_term_true_val_s) + losses.second_derivative_loss_fn(V_term_pred_val_s, V_term_true_val_s)
                
                Voc_pred_val_s = pred_val_scaled[:, :, target_col_map["Estimated_Voc_Kalman"]]
                Voc_true_val_s = yb_val[:, :, target_col_map["Estimated_Voc_Kalman"]]
                val_loss_deriv_Voc = losses.derivative_loss_fn(Voc_pred_val_s, Voc_true_val_s) + losses.second_derivative_loss_fn(Voc_pred_val_s, Voc_true_val_s)
                
                val_loss_deriv = val_loss_deriv_V_term + 0.3 * val_loss_deriv_Voc
                final_val_loss = val_total_loss + config.WEIGHT_DERIVATIVE * val_loss_deriv
                val_loss_sum += final_val_loss.item()
                val_count += 1
        
        avg_val_loss = val_loss_sum / max(1, val_count)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            pat_cnt = 0
            torch.save({'model_state': model.state_dict()}, save_path)
        else:
            pat_cnt += 1
            if pat_cnt >= config.PATIENCE:
                break

    ckpt = torch.load(save_path, map_location=config.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Testing", disable=True):
            xb = xb.to(config.device)
            pred = model(xb).cpu()
            y_true.append(yb)
            y_pred.append(pred)

    y_true_scaled_all = torch.cat(y_true, dim=0).numpy()
    y_pred_scaled_all = torch.cat(y_pred, dim=0).numpy()

    true_flat_orig = target_scaler.inverse_transform(y_true_scaled_all.reshape(-1, len(config.TARGET_COLUMNS)))
    pred_flat_orig = target_scaler.inverse_transform(y_pred_scaled_all.reshape(-1, len(config.TARGET_COLUMNS)))

    timestamps = []
    if test_segs and not test_segs[0].empty:
        for seg in test_segs:
            if seg.empty:
                continue
            t_arr = seg['time'].values
            for i in range(0, len(seg) - config.TIME_STEPS_IN - config.TIME_STEPS_OUT + 1, config.TIME_STEPS_OUT):
                start_idx = i + config.TIME_STEPS_IN
                end_idx = i + config.TIME_STEPS_IN + config.TIME_STEPS_OUT
                if end_idx <= len(t_arr):
                    timestamps.extend(t_arr[start_idx:end_idx])

    result_df = pd.DataFrame()
    num_pred_points = len(pred_flat_orig)
    
    if timestamps:
        if len(timestamps) != num_pred_points:
            timestamps = timestamps[:num_pred_points] if len(timestamps) > num_pred_points else timestamps + [pd.NaT] * (num_pred_points - len(timestamps))
        result_df["Timestamp"] = timestamps

    for i, col_name in enumerate(config.TARGET_COLUMNS):
        result_df[f"True_{col_name}"] = true_flat_orig[:, i]
        result_df[f"Pred_{col_name}"] = pred_flat_orig[:, i]

    csv_path = config.OUTPUT_DIR / "predictions_result.csv"
    result_df.to_csv(csv_path, index=False, encoding="gbk")

    utils.evaluate_metrics(true_flat_orig, pred_flat_orig, config.TARGET_COLUMNS)

    joblib.dump(feature_scaler, config.OUTPUT_DIR / "feature_scaler.pkl")
    joblib.dump(target_scaler, config.OUTPUT_DIR / "target_scaler.pkl")

if __name__ == "__main__":
    main()