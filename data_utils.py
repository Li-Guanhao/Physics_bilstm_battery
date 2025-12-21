import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import functions
import config

class MemoryEfficientDynamicSeqDataset(Dataset):
    def __init__(self, segment_list, feature_columns, target_columns, feature_scaler, target_scaler, in_steps, out_steps, window_step=1, name="Dataset"):
        self.segment_list = segment_list
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.window_step = window_step
        self.name = name
        self.sample_indices = []

        for seg_idx, seg_df in enumerate(segment_list):
            max_start_idx_in_segment = len(seg_df) - self.in_steps - self.out_steps + 1
            if max_start_idx_in_segment <= 0:
                continue
            for i in range(0, max_start_idx_in_segment, self.window_step):
                self.sample_indices.append((seg_idx, i))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, global_idx):
        segment_idx, start_idx_in_segment = self.sample_indices[global_idx]
        selected_segment_df = self.segment_list[segment_idx]
        
        x_window_data = selected_segment_df[self.feature_columns].iloc[
            start_idx_in_segment : start_idx_in_segment + self.in_steps
        ]
        y_window_data = selected_segment_df[self.target_columns].iloc[
            start_idx_in_segment + self.in_steps : start_idx_in_segment + self.in_steps + self.out_steps
        ]

        x_scaled_np = self.feature_scaler.transform(x_window_data)
        y_scaled_np = self.target_scaler.transform(y_window_data)

        x_tensor = torch.from_numpy(x_scaled_np.astype(np.float32))
        y_tensor = torch.from_numpy(y_scaled_np.astype(np.float32))

        return x_tensor, y_tensor

def load_and_preprocess_data():
    df_raw = pd.read_csv(config.DATA_FILE, encoding="gbk")
    df = functions.apply_kalman_filter_to_df(df_raw, "Cell_voltage", "total_current")
    
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    seg_flags = df["time"].diff().dt.total_seconds().fillna(0).gt(178)
    df["seg_id"] = seg_flags.cumsum()

    min_len = config.TIME_STEPS_IN + config.TIME_STEPS_OUT
    segments = [g.reset_index(drop=True) for _, g in df.groupby("seg_id") if len(g) >= min_len]
    
    return segments

def get_data_loaders(segments):
    all_feat_df = pd.concat([seg[config.FEATURE_COLUMNS] for seg in segments], ignore_index=True)
    all_tgt_df = pd.concat([seg[config.TARGET_COLUMNS] for seg in segments], ignore_index=True)

    feature_scaler = MinMaxScaler().fit(all_feat_df)
    target_scaler = MinMaxScaler().fit(all_tgt_df)

    n_total = len(segments)
    train_end = int(config.TRAIN_RATIO * n_total)
    val_end = train_end + int(config.VAL_RATIO * n_total)
    
    train_segs = segments[:train_end]
    val_segs = segments[train_end:val_end]
    test_segs = segments[val_end:]

    train_dataset = MemoryEfficientDynamicSeqDataset(
        segment_list=train_segs,
        feature_columns=config.FEATURE_COLUMNS,
        target_columns=config.TARGET_COLUMNS,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        in_steps=config.TIME_STEPS_IN,
        out_steps=config.TIME_STEPS_OUT,
        window_step=1,
        name="TrainSet"
    )

    val_dataset = MemoryEfficientDynamicSeqDataset(
        segment_list=val_segs,
        feature_columns=config.FEATURE_COLUMNS,
        target_columns=config.TARGET_COLUMNS,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        in_steps=config.TIME_STEPS_IN,
        out_steps=config.TIME_STEPS_OUT,
        window_step=1,
        name="ValidationSet"
    )

    test_dataset = MemoryEfficientDynamicSeqDataset(
        segment_list=test_segs,
        feature_columns=config.FEATURE_COLUMNS,
        target_columns=config.TARGET_COLUMNS,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        in_steps=config.TIME_STEPS_IN,
        out_steps=config.TIME_STEPS_OUT,
        window_step=config.TIME_STEPS_OUT,
        name="TestSet"
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=10, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, feature_scaler, target_scaler, test_segs