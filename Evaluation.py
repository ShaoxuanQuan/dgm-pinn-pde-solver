import torch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from DGMNet import (DGMNet)

def black_scholes_call_exact(S, t, K, T, r, sigma):

    S = np.asarray(S)
    t = np.asarray(t)

    price = np.zeros_like(S, dtype=float)

    at_maturity_indices = np.where(np.abs(t - T) < 1e-10)
    if at_maturity_indices[0].size > 0:
        price[at_maturity_indices] = np.maximum(S[at_maturity_indices] - K, 0)

    before_maturity_indices = np.where(t < T)
    if before_maturity_indices[0].size > 0:

        S_b = S[before_maturity_indices]
        t_b = t[before_maturity_indices]

        epsilon = 1e-10
        S_b = np.maximum(S_b, epsilon)
        time_to_maturity = T - t_b

        d1 = (np.log(S_b / K) + (r + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
        d2 = d1 - sigma * np.sqrt(time_to_maturity)

        price_b = S_b * norm.cdf(d1) - K * np.exp(-r * time_to_maturity) * norm.cdf(d2)
        price[before_maturity_indices] = price_b

    return price

FIN_PARAMS = {
    'r': 0.05, 'sigma': 0.2, 'K': 100.0, 'T': 1.0,
    'S_min': 0.0,
    'S_max_interest': 200.0,
    'S_max_training': 300.0
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = 'dgm_bs_model.pth'

lb = torch.tensor([0.0, FIN_PARAMS['S_min']], dtype=torch.float32, device=DEVICE)
ub = torch.tensor([FIN_PARAMS['T'], FIN_PARAMS['S_max_training']], dtype=torch.float32, device=DEVICE)
dgm_model = DGMNet(input_dim=2, hidden_dim=24, num_layers=6, lb=lb, ub=ub).to(DEVICE)

dgm_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

print(f"Model state loaded from {MODEL_SAVE_PATH}")


print("\n" + "="*60)
print("           Starting Final Model Performance Evaluation")

dgm_model.eval()

S_eval_np = np.linspace(1e-6, FIN_PARAMS['S_max_interest'], 500)

time_points_eval = np.linspace(0, FIN_PARAMS['T'] * 0.99, 10) # 评估10个时间切片

S_grid, T_grid = np.meshgrid(S_eval_np, time_points_eval)

S_flat = S_grid.flatten()
T_flat = T_grid.flatten()
eval_points_np = np.stack([T_flat, S_flat], axis=1)
eval_points_tensor = torch.from_numpy(eval_points_np).float().to(DEVICE)

with torch.no_grad():

    V_pred_np = dgm_model(eval_points_tensor).cpu().numpy().flatten()

V_exact_np = black_scholes_call_exact(
    S=S_flat, t=T_flat,
    K=FIN_PARAMS['K'], T=FIN_PARAMS['T'],
    r=FIN_PARAMS['r'], sigma=FIN_PARAMS['sigma']
)

abs_error = np.abs(V_pred_np - V_exact_np)
mean_abs_error = np.mean(abs_error)
max_abs_error = np.max(abs_error)
rmse = np.sqrt(np.mean(abs_error**2))

epsilon = 1e-8
relative_error = abs_error / (V_exact_np + epsilon)
mean_relative_error_percentage = np.mean(relative_error) * 100

l2_error_norm = np.linalg.norm(V_pred_np - V_exact_np)
l2_exact_norm = np.linalg.norm(V_exact_np)
relative_l2_error = l2_error_norm / l2_exact_norm

range_exacts = np.max(V_exact_np) - np.min(V_exact_np)
nrmse_range = rmse / range_exacts if range_exacts > epsilon else 0

mean_exacts = np.mean(V_exact_np)
nrmse_mean = rmse / mean_exacts if abs(mean_exacts) > epsilon else 0

print(f"Evaluation Grid: {len(S_eval_np)} S-points x {len(time_points_eval)} t-points = {len(eval_points_np)} total points.")
print(f"Evaluation S-range: [{S_eval_np[0]:.2f}, {S_eval_np[-1]:.2f}]")
print(f"Evaluation T-range: [{time_points_eval[0]:.2f}, {time_points_eval[-1]:.2f}]")
print("-" * 60)

print("--- Absolute Error Metrics ---")
print(f"  - Root Mean Squared Error (RMSE):    {rmse:.6f}")
print(f"  - Mean Absolute Error (MAE):         {mean_abs_error:.6f}")
print(f"  - Maximum Absolute Error (Max Error):{max_abs_error:.6f}")
print("-" * 60)

print("--- Relative Error Metrics ---")
print(f"  - Relative L2 Error:               {relative_l2_error:.4%} ({relative_l2_error:.6f})")
print(f"  - Mean Relative Error:             {mean_relative_error_percentage:.2f}% (Note: can be skewed by small denominators)")
print("-" * 60)

print("--- Normalized RMSE (NRMSE) Metrics ---")
print(f"  - Exact Values Range (max-min):    {range_exacts:.4f}")
print(f"  - NRMSE (normalized by range):     {nrmse_range:.4%} ({nrmse_range:.6f})")
print(f"  - Exact Values Mean:               {mean_exacts:.4f}")
print(f"  - NRMSE (normalized by mean):      {nrmse_mean:.4%} ({nrmse_mean:.6f})")
print("="*60)