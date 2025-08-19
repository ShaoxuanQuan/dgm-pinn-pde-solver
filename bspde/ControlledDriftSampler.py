import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import compute_residual_and_grads, compute_residual_and_grads_2D

class ControlledDriftSampler:

    def __init__(self, pde_params: dict, T: float, s_min: float, s_max: float,
                 batch_size: int, sampling_std: float = 30.0, device='cuda',
                 scaling_factor: float = 0.01, u_max: float = 1.0,
                 n_range: tuple = (15, 30),
                 boundary_memory_size: int = 2048):

        self.pde_params = pde_params
        self.r = pde_params['r']
        self.sigma = pde_params['sigma']
        self.K = pde_params['K']

        self.T = T
        self.s_min = s_min
        self.s_max = s_max
        self.device = device
        self.batch_size = batch_size

        self.sampling_std = sampling_std

        self.scaling_factor = scaling_factor
        self.u_max = u_max

        self.alpha = 1.0

        self.n_range = n_range 
        self.dt = self._initialize_dt()  

        self.current_points = self._initialize_points()

        self.boundary_memory = self._initialize_points(batch_size=boundary_memory_size)
        self.boundary_memory[:, 0:1] = self.T 

    def _initialize_dt(self):

        N = torch.randint(self.n_range[0], self.n_range[1] + 1, size=(self.batch_size, 1), device=self.device)

        dt = self.T / N
        return dt

    def _initialize_points(self, batch_size=None) -> torch.Tensor:
        if batch_size is None:
            batch_size = self.batch_size

        S_initial = torch.randn(batch_size, 1, device=self.device) * self.sampling_std + self.K
        S_initial = torch.clamp(S_initial, self.s_min, self.s_max)
        t_initial = torch.zeros(batch_size, 1, device=self.device) 
        points = torch.cat([t_initial, S_initial], dim=1)
        return points

    def _sde_step(self, S, u, dt):
        dW = torch.randn_like(S) * torch.sqrt(dt) 
        controlled_drift = self.r * S + u
        S_new = S + controlled_drift * dt + self.sigma * S * dW
        return torch.clamp(S_new, self.s_min, self.s_max)

    def sample_batch(self, model: torch.nn.Module):
        points_for_training = self.current_points.clone().detach().requires_grad_(True)

        residual_dict = compute_residual_and_grads(points_for_training, model, self.pde_params)
        residual_s = residual_dict['residual_s']
        u = self.scaling_factor * residual_s.detach()
        u = torch.clamp(u, -self.u_max, self.u_max)

        with torch.no_grad():
            t_current = self.current_points[:, 0:1]
            S_current = self.current_points[:, 1:2]

            scaled_u = (1.0 - self.alpha) * u

            S_next = self._sde_step(S_current, scaled_u, self.dt)
            t_next = t_current + self.dt

            expired_mask = (t_next.squeeze() >= self.T)
            num_expired = expired_mask.sum().item()

            if num_expired > 0:
                expired_points = torch.cat([
                    torch.full_like(S_next[expired_mask], self.T), 
                    S_next[expired_mask]
                ], dim=1)

                self.boundary_memory = torch.cat([expired_points, self.boundary_memory[:-num_expired]], dim=0)

                reinitialized_points = self._initialize_points(batch_size=num_expired)
                S_next[expired_mask] = reinitialized_points[:, 1:2]
                t_next[expired_mask] = reinitialized_points[:, 0:1]

                self.dt[expired_mask] = self._initialize_dt()[expired_mask]

            self.current_points = torch.cat([t_next, S_next], dim=1)

        return points_for_training

    def update_alpha(self, decay_rate, min_alpha=0.0):
        self.alpha = max(min_alpha, self.alpha - decay_rate)

'''======================================================================================='''


class ControlledDriftSampler_2D:

    def __init__(self, pde_params: dict, T: float, s_min: tuple, s_max: tuple,
                 batch_size: int, sampling_std: tuple, device='cuda',
                 scaling_factor: float = 0.01, u_max: float = 1.0,
                 n_range: tuple = (20, 40),
                 boundary_memory_size: int = 4096):

        self.pde_params = pde_params
        self.r = pde_params['r']
        self.sigma1 = pde_params['sigma1']
        self.sigma2 = pde_params['sigma2']
        self.rho = pde_params['rho']
        self.K = pde_params['K']

        self.T = T
        self.s_min = torch.tensor(s_min, device=device).view(1, 2)
        self.s_max = torch.tensor(s_max, device=device).view(1, 2)
        self.sampling_std = torch.tensor(sampling_std, device=device).view(1, 2)
        self.K_tensor = torch.tensor([self.K, self.K], device=device).view(1, 2)

        self.device = device
        self.batch_size = batch_size

        self.scaling_factor = scaling_factor
        self.u_max = u_max

        self.alpha = 1.0

        self.n_range = n_range
        self.dt = self._initialize_dt()

        rho_tensor = torch.tensor(self.rho, device=self.device)
        L_22 = torch.sqrt(1.0 - rho_tensor ** 2)
        self.L = torch.tensor([
            [1.0, 0.0],
            [rho_tensor, L_22]
        ], device=self.device)

        self.current_points = self._initialize_points()

        self.boundary_memory = self._initialize_points(batch_size=boundary_memory_size)
        self.boundary_memory[:, 0:1] = self.T

    def _initialize_dt(self):
        N = torch.randint(self.n_range[0], self.n_range[1] + 1, size=(self.batch_size, 1), device=self.device)
        dt = self.T / N
        return dt

    def _initialize_points(self, batch_size=None) -> torch.Tensor:
        if batch_size is None:
            batch_size = self.batch_size

        S_initial = torch.randn(batch_size, 2, device=self.device) * self.sampling_std + self.K_tensor
        S_initial = torch.max(torch.min(S_initial, self.s_max), self.s_min) 
        t_initial = torch.zeros(batch_size, 1, device=self.device)
        # points 是 (batch_size, 3)
        points = torch.cat([t_initial, S_initial], dim=1)
        return points

    def _sde_step(self, S_vec, u_vec, dt):
        sqrt_dt = torch.sqrt(dt)
        sqrt_dt_reshaped = sqrt_dt.unsqueeze(-1)
        dZ = torch.randn(self.batch_size, 2, 1, device=self.device) * sqrt_dt_reshaped
        dW = torch.matmul(self.L, dZ).squeeze(-1)

        drift = self.r * S_vec + u_vec
        volatility_term_efficient = S_vec * torch.tensor([self.sigma1, self.sigma2], device=self.device) * dW

        S_new = S_vec + drift * dt + volatility_term_efficient

        return torch.max(torch.min(S_new, self.s_max), self.s_min)

    def sample_batch(self, model: torch.nn.Module):
        points_for_training = self.current_points.clone().detach().requires_grad_(True)

        residual_dict = compute_residual_and_grads_2D(points_for_training, model, self.pde_params)

        residual_s1 = residual_dict['residual_s1']
        residual_s2 = residual_dict['residual_s2']

        u = torch.cat([residual_s1, residual_s2], dim=1)
        u = self.scaling_factor * u.detach()
        u = torch.clamp(u, -self.u_max, self.u_max)

        with torch.no_grad():
            t_current = self.current_points[:, 0:1]
            S_current = self.current_points[:, 1:]  # S_current is (batch_size, 2)

            scaled_u = (1.0 - self.alpha) * u

            S_next = self._sde_step(S_current, scaled_u, self.dt)
            t_next = t_current + self.dt

            expired_mask = (t_next.squeeze() >= self.T)
            num_expired = expired_mask.sum().item()

            if num_expired > 0:
                expired_points = torch.cat([
                    torch.full_like(t_next[expired_mask], self.T),
                    S_next[expired_mask]
                ], dim=1)

                self.boundary_memory = torch.cat([expired_points, self.boundary_memory[:-num_expired]], dim=0)

                reinitialized_points = self._initialize_points(batch_size=num_expired)
                S_next[expired_mask] = reinitialized_points[:, 1:]
                t_next[expired_mask] = reinitialized_points[:, 0:1]

                self.dt[expired_mask] = self._initialize_dt()[expired_mask]

            self.current_points = torch.cat([t_next, S_next], dim=1)

        return points_for_training

    def update_alpha(self, decay_rate, min_alpha=0.0):

        self.alpha = max(min_alpha, self.alpha - decay_rate)
