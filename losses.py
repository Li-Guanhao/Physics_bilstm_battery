import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    def __init__(self, target_min_vals, target_scale_vals, target_col_indices, lambda_physics=0.5, lambda_aux=0.1, lambda_main=20.0):
        super().__init__()
        self.target_min_vals = target_min_vals
        self.target_scale_vals = target_scale_vals
        self.idx_V_terminal = target_col_indices['V_terminal']
        self.idx_I_future = target_col_indices['I_future']
        self.idx_Voc = target_col_indices['Voc']
        self.idx_Vp = target_col_indices['Vp']
        self.idx_Vo = target_col_indices['Vo']
        self.idx_Ro = target_col_indices['Ro']
        
        self.lambda_physics = lambda_physics
        self.lambda_aux = lambda_aux
        self.lambda_main = lambda_main
        self.mse_loss = nn.L1Loss()

    def _inverse_transform_tensor(self, scaled_tensor, target_idx):
        min_val = self.target_min_vals[target_idx].view(1, 1, -1) if scaled_tensor.ndim == 3 else self.target_min_vals[target_idx]
        scale_val = self.target_scale_vals[target_idx].view(1, 1, -1) if scaled_tensor.ndim == 3 else self.target_scale_vals[target_idx]
        scale_val = torch.where(scale_val == 0, torch.tensor(1.0, device=scale_val.device), scale_val)
        return scaled_tensor * scale_val + min_val

    def forward(self, pred_scaled, true_scaled):
        V_terminal_pred_s = pred_scaled[:, :, self.idx_V_terminal]
        V_terminal_true_s = true_scaled[:, :, self.idx_V_terminal]
        loss_main_pred = self.mse_loss(V_terminal_pred_s, V_terminal_true_s)

        I_future_pred_s = pred_scaled[:, :, self.idx_I_future]
        I_future_true_s = true_scaled[:, :, self.idx_I_future]
        loss_aux_I = self.mse_loss(I_future_pred_s, I_future_true_s)

        Voc_pred_s = pred_scaled[:, :, self.idx_Voc]
        Voc_true_s = true_scaled[:, :, self.idx_Voc]
        loss_aux_voc = self.mse_loss(Voc_pred_s, Voc_true_s)

        Vp_pred_s = pred_scaled[:, :, self.idx_Vp]
        Vp_true_s = true_scaled[:, :, self.idx_Vp]
        loss_aux_vp = self.mse_loss(Vp_pred_s, Vp_true_s)

        Vo_direct_pred_s = pred_scaled[:, :, self.idx_Vo]
        Vo_direct_true_s = true_scaled[:, :, self.idx_Vo]
        loss_aux_vo_direct = self.mse_loss(Vo_direct_pred_s, Vo_direct_true_s)
        
        Ro_pred_s = pred_scaled[:, :, self.idx_Ro]
        Ro_true_s = true_scaled[:, :, self.idx_Ro]
        loss_aux_ro = self.mse_loss(Ro_pred_s, Ro_true_s)

        total_aux_loss = loss_aux_I + loss_aux_voc + loss_aux_vp + loss_aux_ro + loss_aux_vo_direct

        V_terminal_pred_orig = self._inverse_transform_tensor(V_terminal_pred_s, self.idx_V_terminal)
        Voc_pred_orig = self._inverse_transform_tensor(Voc_pred_s, self.idx_Voc)
        Vp_pred_orig = self._inverse_transform_tensor(Vp_pred_s, self.idx_Vp)
        Vo_direct_pred_orig = self._inverse_transform_tensor(Vo_direct_pred_s, self.idx_Vo)
        I_future_pred_orig = self._inverse_transform_tensor(I_future_pred_s, self.idx_I_future)
        Ro_pred_orig = self._inverse_transform_tensor(Ro_pred_s, self.idx_Ro)

        Vo_calculated_for_Vterminal_orig = Ro_pred_orig * I_future_pred_orig
        V_terminal_physics_calc1_orig = Voc_pred_orig - Vp_pred_orig - Vo_calculated_for_Vterminal_orig
        loss_physics1 = self.mse_loss(V_terminal_pred_orig, V_terminal_physics_calc1_orig)

        Vo_physics_calc2_orig = Ro_pred_orig * I_future_pred_orig
        Vo_true_orig = self._inverse_transform_tensor(Vo_direct_true_s, self.idx_Vo)
        loss_physics2 = self.mse_loss(Vo_physics_calc2_orig, Vo_true_orig)

        total_physics_loss = loss_physics1 + loss_physics2

        total_loss = self.lambda_main * loss_main_pred + self.lambda_aux * total_aux_loss + self.lambda_physics * total_physics_loss
        return total_loss, self.lambda_main * loss_main_pred, total_aux_loss, total_physics_loss

def derivative_loss_fn(pred_main_scaled, true_main_scaled):
    if pred_main_scaled.ndim == 2:
        dp = pred_main_scaled[:, 1:] - pred_main_scaled[:, :-1]
        dt = true_main_scaled[:, 1:] - true_main_scaled[:, :-1]
    elif pred_main_scaled.ndim == 3 and pred_main_scaled.size(2) == 1:
        dp = pred_main_scaled[:, 1:, 0] - pred_main_scaled[:, :-1, 0]
        dt = true_main_scaled[:, 1:, 0] - true_main_scaled[:, :-1, 0]
    else:
        raise ValueError(f"derivative_loss_fn input shape not handled: {pred_main_scaled.shape}")
    return ((dp - dt)**2).mean()

def second_derivative_loss_fn(pred, true):
    dp1 = pred[:, 1:] - pred[:, :-1]
    dt1 = true[:, 1:] - true[:, :-1]
    dp2 = dp1[:, 1:] - dp1[:, :-1]
    dt2 = dt1[:, 1:] - dt1[:, :-1]
    return ((dp2 - dt2)**2).mean()