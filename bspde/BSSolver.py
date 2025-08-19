import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from utils import compute_residual_and_grads, compute_residual_and_grads_2D

class BSSolver:

    def __init__(self, model: nn.Module, pde_params: dict, terminal_condition: dict,
                 optimizer: torch.optim.Optimizer, scheduler: StepLR,
                 pde_sampler, terminal_sampler,
                 lambda_terminal: float, lambda_g_pde: float, device: str = 'cpu'):
        self.model = model.to(device)
        self.pde_params = pde_params
        self.terminal_config = terminal_condition
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_sampler = pde_sampler 
        self.terminal_sampler = terminal_sampler
        self.lambda_terminal = lambda_terminal
        self.lambda_g_pde = lambda_g_pde
        self.device = device

    def _compute_pde_loss(self, pde_points: torch.Tensor) -> torch.Tensor:
        residual_dict = compute_residual_and_grads(pde_points, self.model, self.pde_params)
        residual = residual_dict['residual']
        residual_t = residual_dict['residual_t']
        residual_s = residual_dict['residual_s']
        loss_f = torch.mean(residual ** 2)
        loss_g = torch.mean(residual_t ** 2) + torch.mean(residual_s ** 2)
        return loss_f + self.lambda_g_pde * loss_g

    def _compute_terminal_loss(self, terminal_points: torch.Tensor) -> torch.Tensor: 
        tc = self.terminal_config
        s = terminal_points[:, 1:2]

        true_values = torch.relu(s - tc['K'])
        pred_values = self.model(terminal_points)

        return torch.mean((pred_values - true_values) ** 2)

    def solve(self, num_iterations: int, alpha_decay_rate: float): 
        loss_history = []
        pbar = tqdm(range(num_iterations), desc="Solving BS PDE (gPINN)", ncols=120)

        for i in pbar:
            self.model.train()
            self.optimizer.zero_grad()

            pde_points = self.pde_sampler.sample_batch(self.model) 
            terminal_points = self.terminal_sampler.sample_batch() 

            loss_pde = self._compute_pde_loss(pde_points)
            loss_terminal = self._compute_terminal_loss(terminal_points)

            total_loss = loss_pde + self.lambda_terminal * loss_terminal
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.pde_sampler.update_alpha(alpha_decay_rate)

            if hasattr(self.pde_sampler, 'update'):
                self.pde_sampler.update(loss_pde.item())

            loss_dict = {
                'total_loss': total_loss.item(),
                'pde_loss': loss_pde.item(),
                'terminal_loss': loss_terminal.item()
            }
            loss_history.append(loss_dict)

            if (i + 1) % 100 == 0:
                pbar.set_postfix(
                    loss=f"{loss_dict['total_loss']:.2e}",
                    alpha=f"{self.pde_sampler.alpha:.3f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

        print("Training finished.")
        return loss_history

'''======================================================================================='''

class BSSolver_2D:

    def __init__(self, model: nn.Module, pde_params: dict, terminal_condition: dict,
                 optimizer: torch.optim.Optimizer, scheduler: StepLR,
                 pde_sampler, terminal_sampler,
                 lambda_terminal: float, lambda_g_pde: float, device: str = 'cpu'):
        self.model = model.to(device)
        self.pde_params = pde_params
        self.terminal_config = terminal_condition
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_sampler = pde_sampler
        self.terminal_sampler = terminal_sampler
        self.lambda_terminal = lambda_terminal
        self.lambda_g_pde = lambda_g_pde
        self.device = device

    def _compute_pde_loss(self, pde_points: torch.Tensor) -> torch.Tensor:

        residual_dict = compute_residual_and_grads_2D(pde_points, self.model, self.pde_params)

        residual = residual_dict['residual']
        residual_t = residual_dict['residual_t']
        residual_s1 = residual_dict['residual_s1']
        residual_s2 = residual_dict['residual_s2']

        loss_f = torch.mean(residual ** 2)

        loss_g = torch.mean(residual_t ** 2) + torch.mean(residual_s1 ** 2) + torch.mean(residual_s2 ** 2)

        return loss_f + self.lambda_g_pde * loss_g

    def _compute_terminal_loss(self, terminal_points: torch.Tensor) -> torch.Tensor:
        tc = self.terminal_config

        s1 = terminal_points[:, 1:2]
        s2 = terminal_points[:, 2:3]

        w1 = tc['w1']
        w2 = tc['w2']
        K = tc['K']

        true_values = torch.relu(w1 * s1 + w2 * s2 - K)
        pred_values = self.model(terminal_points)

        return torch.mean((pred_values - true_values) ** 2)

    def solve(self, num_iterations: int, alpha_decay_rate: float):
        loss_history = []
        pbar = tqdm(range(num_iterations), desc="Solving 2D BS PDE (gPINN)", ncols=120)

        for i in pbar:
            self.model.train()
            self.optimizer.zero_grad()

            pde_points = self.pde_sampler.sample_batch(self.model)
            terminal_points = self.terminal_sampler.sample_batch()

            loss_pde = self._compute_pde_loss(pde_points)
            loss_terminal = self._compute_terminal_loss(terminal_points)

            total_loss = loss_pde + self.lambda_terminal * loss_terminal
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.pde_sampler.update_alpha(alpha_decay_rate)
            if hasattr(self.pde_sampler, 'update'):
                self.pde_sampler.update(loss_pde.item())

            loss_dict = {
                'total_loss': total_loss.item(),
                'pde_loss': loss_pde.item(),
                'terminal_loss': loss_terminal.item()
            }
            loss_history.append(loss_dict)

            if (i + 1) % 100 == 0:
                pbar.set_postfix(
                    loss=f"{loss_dict['total_loss']:.2e}",
                    alpha=f"{self.pde_sampler.alpha:.3f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

        print("Training finished.")
        return loss_history

