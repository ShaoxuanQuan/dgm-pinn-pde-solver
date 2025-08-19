import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from DGMNet import DGMNet
from BSSolver import BSSolver
from ControlledDriftSampler import ControlledDriftSampler
from utils import compute_residual_and_grads

class TerminalSampler:

    def __init__(self, pde_sampler: ControlledDriftSampler, batch_size: int):
        self.pde_sampler = pde_sampler
        self.batch_size = batch_size

    def sample_batch(self):
        memory_size = self.pde_sampler.boundary_memory.shape[0]
        indices = torch.randint(0, memory_size, (self.batch_size,))

        return self.pde_sampler.boundary_memory[indices].clone().detach().requires_grad_(True)


# === 步骤 2: 主脚本 ===
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    FIN_PARAMS = {
        'r': 0.05, 'sigma': 0.2, 'K': 100.0, 'T': 1.0,
        'S_min': 0.0,
        'S_max_interest': 200.0,
        'S_max_training': 300.0
    }

    TRAIN_PARAMS = {
        'lr': 5e-3, 'lr_decay_step': 100, 'lr_decay_gamma': 0.99,
        'num_iterations': 10000, 'pde_batch_size': 4000,
        'initial_sampling_std': 30.0,
        'terminal_batch_size': 2000,
        'lambda_terminal': 2.0,
        'lambda_g_pde': 0.1,
        'alpha_decay_rate': 1.0 / 8000,
        'sampler_scaling_factor': 0.1,
        'sampler_u_max': 1.0,
        'sde_n_range': (20, 40),
        'boundary_memory_size': 4096
    }

    lb = torch.tensor([0.0, FIN_PARAMS['S_min']], dtype=torch.float32, device=DEVICE)
    ub = torch.tensor([FIN_PARAMS['T'], FIN_PARAMS['S_max_training']], dtype=torch.float32, device=DEVICE)

    dgm_model = DGMNet(input_dim=2, hidden_dim=24, num_layers=6, lb=lb, ub=ub).to(DEVICE)  # 确保模型在正确的设备上

    optimizer = torch.optim.Adam(dgm_model.parameters(), lr=TRAIN_PARAMS['lr'])
    scheduler = StepLR(optimizer, step_size=TRAIN_PARAMS['lr_decay_step'], gamma=TRAIN_PARAMS['lr_decay_gamma'])

    # --- 核心修改：实例化采样器 ---

    # 1. 创建 pde_sampler (生产者)
    pde_sampler = ControlledDriftSampler(
        pde_params=FIN_PARAMS,
        T=FIN_PARAMS['T'],
        s_min=FIN_PARAMS['S_min'],
        s_max=FIN_PARAMS['S_max_training'],
        batch_size=TRAIN_PARAMS['pde_batch_size'],
        sampling_std=TRAIN_PARAMS['initial_sampling_std'],
        device=DEVICE,
        scaling_factor=TRAIN_PARAMS['sampler_scaling_factor'],
        u_max=TRAIN_PARAMS['sampler_u_max'],
        # 传递新参数
        n_range=TRAIN_PARAMS['sde_n_range'],
        boundary_memory_size=TRAIN_PARAMS['boundary_memory_size']
    )

    terminal_sampler = TerminalSampler(
        pde_sampler=pde_sampler,
        batch_size=TRAIN_PARAMS['terminal_batch_size']
    )

    terminal_condition_config = {
        'T': FIN_PARAMS['T'], 'K': FIN_PARAMS['K'],
        'S_min': FIN_PARAMS['S_min'], 'S_max': FIN_PARAMS['S_max_training']
    }

    solver = BSSolver(
        model=dgm_model,
        pde_params=FIN_PARAMS,
        terminal_condition=terminal_condition_config,
        optimizer=optimizer,
        scheduler=scheduler,
        pde_sampler=pde_sampler,
        terminal_sampler=terminal_sampler,
        lambda_terminal=TRAIN_PARAMS['lambda_terminal'],
        lambda_g_pde=TRAIN_PARAMS['lambda_g_pde'],
        device=DEVICE
    )

    solver.solve(
        num_iterations=TRAIN_PARAMS['num_iterations'],
        alpha_decay_rate=TRAIN_PARAMS['alpha_decay_rate']
    )

    MODEL_SAVE_PATH = 'dgm_bs_model.pth'

    torch.save(dgm_model.state_dict(), MODEL_SAVE_PATH)

    print("\n" + "=" * 50)
    print("           Training Complete!")
    print(f"   Model state dictionary saved to: {MODEL_SAVE_PATH}")
    print("=" * 50)
