import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from DGMNet import DGMNet
from BSSolver import BSSolver_2D
from ControlledDriftSampler import ControlledDriftSampler_2D
from utils import compute_residual_and_grads_2D


class TerminalSampler_2D:
    def __init__(self, pde_sampler: ControlledDriftSampler_2D, batch_size: int):
        self.pde_sampler = pde_sampler
        self.batch_size = batch_size

    def sample_batch(self):
        memory_size = self.pde_sampler.boundary_memory.shape[0]
        indices = torch.randint(0, memory_size, (self.batch_size,))
        return self.pde_sampler.boundary_memory[indices].clone().detach().requires_grad_(True)


# === 步骤 2: 主脚本 Script_2D ===
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- 更新金融参数以适应二维问题 ---
    FIN_PARAMS = {
        'r': 0.05,
        'sigma1': 0.2,  # 资产1的波动率
        'sigma2': 0.25,  # 资产2的波动率
        'rho': 0.5,  # 资产间的相关系数
        'K': 100.0,  # 篮子期权的执行价
        'w1': 0.5,  # 资产1在篮子中的权重
        'w2': 0.5,  # 资产2在篮子中的权重
        'T': 1.0,
        'S1_min': 0.0, 'S1_max': 300.0,
        'S2_min': 0.0, 'S2_max': 300.0
    }

    TRAIN_PARAMS = {
        'lr': 5e-3,  # 从一个更安全的值开始
        'lr_decay_step': 100,
        'lr_decay_gamma': 0.99,
        'num_iterations': 10000,
        'pde_batch_size': 4096,
        'terminal_batch_size': 2048,
        'lambda_terminal': 2.0,
        'lambda_g_pde': 0.1,
        'alpha_decay_rate': 1.0 / 8000,  # 匹配新的迭代次数
        'sampler_scaling_factor': 0.05,  # 二维问题中控制项可能更敏感，稍微调低
        'sampler_u_max': 1.0,
        'sde_n_range': (20, 40),
        'boundary_memory_size': 4096,
        # 二维初始采样标准差 (std_s1, std_s2)
        'initial_sampling_std': (30.0, 30.0)
    }

    # --- 更新模型和采样器实例化 ---

    # lb 和 ub 现在是三维的
    lb = torch.tensor([0.0, FIN_PARAMS['S1_min'], FIN_PARAMS['S2_min']], dtype=torch.float32, device=DEVICE)
    ub = torch.tensor([FIN_PARAMS['T'], FIN_PARAMS['S1_max'], FIN_PARAMS['S2_max']], dtype=torch.float32, device=DEVICE)

    # DGMNet 的 input_dim 必须是 3
    dgm_model = DGMNet(input_dim=3, hidden_dim=64, num_layers=6, lb=lb, ub=ub).to(DEVICE)  # 建议增加隐藏层维度

    optimizer = torch.optim.Adam(dgm_model.parameters(), lr=TRAIN_PARAMS['lr'])
    scheduler = StepLR(optimizer, step_size=TRAIN_PARAMS['lr_decay_step'], gamma=TRAIN_PARAMS['lr_decay_gamma'])

    # 1. 创建 pde_sampler_2d
    pde_sampler_2d = ControlledDriftSampler_2D(
        pde_params=FIN_PARAMS,
        T=FIN_PARAMS['T'],
        s_min=(FIN_PARAMS['S1_min'], FIN_PARAMS['S2_min']),  # 传递元组
        s_max=(FIN_PARAMS['S1_max'], FIN_PARAMS['S2_max']),  # 传递元组
        batch_size=TRAIN_PARAMS['pde_batch_size'],
        sampling_std=TRAIN_PARAMS['initial_sampling_std'],  # 传递元组
        device=DEVICE,
        scaling_factor=TRAIN_PARAMS['sampler_scaling_factor'],
        u_max=TRAIN_PARAMS['sampler_u_max'],
        n_range=TRAIN_PARAMS['sde_n_range'],
        boundary_memory_size=TRAIN_PARAMS['boundary_memory_size']
    )

    # 2. 创建 terminal_sampler_2d
    terminal_sampler_2d = TerminalSampler_2D(
        pde_sampler=pde_sampler_2d,
        batch_size=TRAIN_PARAMS['terminal_batch_size']
    )

    # terminal_condition_config 现在需要包含权重 w1, w2
    terminal_condition_config = {
        'T': FIN_PARAMS['T'], 'K': FIN_PARAMS['K'],
        'w1': FIN_PARAMS['w1'], 'w2': FIN_PARAMS['w2']
    }

    # 3. 实例化 BSSolver_2D
    solver = BSSolver_2D(
        model=dgm_model,
        pde_params=FIN_PARAMS,
        terminal_condition=terminal_condition_config,
        optimizer=optimizer,
        scheduler=scheduler,
        pde_sampler=pde_sampler_2d,
        terminal_sampler=terminal_sampler_2d,
        lambda_terminal=TRAIN_PARAMS['lambda_terminal'],
        lambda_g_pde=TRAIN_PARAMS['lambda_g_pde'],
        device=DEVICE
    )

    # 4. 运行求解器
    solver.solve(
        num_iterations=TRAIN_PARAMS['num_iterations'],
        alpha_decay_rate=TRAIN_PARAMS['alpha_decay_rate']
    )

    MODEL_SAVE_PATH = 'dgm_bs_model_2d.pth'
    torch.save(dgm_model.state_dict(), MODEL_SAVE_PATH)

    print("\n" + "=" * 50)
    print("       2D Training Complete!")
    print(f"   Model state dictionary saved to: {MODEL_SAVE_PATH}")
    print("=" * 50)



