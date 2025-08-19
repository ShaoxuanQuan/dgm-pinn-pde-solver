import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class ControlledDriftSamplerHJB:

    def __init__(self, hjb_config, T, x_domain, batch_size, device='cuda',
                 n_range=(15, 30), boundary_memory_size=2048):

        self.hjb = hjb_config
        self.T = T
        self.x_min, self.x_max = x_domain
        self.device = device
        self.batch_size = batch_size
        self.n_range = n_range

        self.dt = self._initialize_dt()

        self.current_points = self._initialize_points()

        self.boundary_memory = self._initialize_points(batch_size=boundary_memory_size)
        self.boundary_memory[:, 0:1] = self.T 

        self.u = torch.zeros(self.batch_size, self.hjb.sol_dim, device=self.device)

    def _initialize_dt(self):

        N = torch.randint(self.n_range[0], self.n_range[1] + 1, size=(self.batch_size, 1), device=self.device)
        return self.T / N

    def _initialize_points(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        x_initial = torch.rand(batch_size, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        t_initial = torch.zeros(batch_size, 1, device=self.device)

        return torch.cat([t_initial, x_initial], dim=1)

    def _sde_step(self, x, u, dt):

        mu, r, sigma = self.hjb.μ, self.hjb.r, self.hjb.σ

        drift = ((mu - r) * u + r) * x
        diffusion = sigma * u * x

        dW = torch.randn_like(x) * torch.sqrt(dt)
        x_new = x + drift * dt + diffusion * dW
        return torch.clamp(x_new, self.x_min, self.x_max)

    def sample_batch(self):

        return self.current_points.clone().detach().requires_grad_(True)

    def update(self, u_from_actor: torch.Tensor):

        with torch.no_grad():
            self.u = u_from_actor.detach()

            t_current = self.current_points[:, 0:1]
            x_current = self.current_points[:, 1:] 

            x_next = self._sde_step(x_current, self.u, self.dt)
            t_next = t_current + self.dt

            expired_mask = (t_next.squeeze() >= self.T)
            num_expired = expired_mask.sum().item()

            if num_expired > 0:
                expired_points = torch.cat([
                    torch.full_like(x_next[expired_mask], self.T),
                    x_next[expired_mask]
                ], dim=1)

                self.boundary_memory = torch.cat([expired_points, self.boundary_memory[:-num_expired]], dim=0)

                reinitialized_points = self._initialize_points(batch_size=num_expired)
                t_next[expired_mask] = reinitialized_points[:, 0:1]
                x_next[expired_mask] = reinitialized_points[:, 1:]

                self.dt[expired_mask] = self._initialize_dt()[:num_expired]

            self.current_points = torch.cat([t_next, x_next], dim=1)


class TerminalSampler:

    def __init__(self, domain_sampler: ControlledDriftSamplerHJB, batch_size: int):
        self.domain_sampler = domain_sampler
        self.batch_size = batch_size

    def sample_batch(self):
        memory_size = self.domain_sampler.boundary_memory.shape[0]
        indices = torch.randint(0, memory_size, (self.batch_size,), device=self.domain_sampler.device)

        return self.domain_sampler.boundary_memory[indices].clone().detach().requires_grad_(True)
