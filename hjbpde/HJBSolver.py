import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from DGMNet import DGMNet, ResNet
from torch.autograd import grad
import os
from ControlledDriftSamplerHJB import ControlledDriftSamplerHJB, TerminalSampler
from collections import OrderedDict

def div(u, points, i=0):

    grad_outputs = torch.ones_like(u)
    gradient = grad(u, points, grad_outputs=grad_outputs, create_graph=True)[0]

    return gradient[:, i:i + 1]


def laplacian(u, points, start_dim=1):

    laplacian_sum = 0
    num_dims = points.shape[1]

    for i in range(start_dim, num_dims):
        first_deriv = div(u, points, i=i)
        second_deriv = div(first_deriv, points, i=i)
        laplacian_sum += second_deriv

    return laplacian_sum

class RiskyAssetConfig:
    def __init__(self, params):
        self.μ, self.σ, self.r, self.γ, self.T = params['mu'], params['sigma'], params['r'], params['gamma'], \
            params['T']
        self.x_domain = (params['x_min'], params['x_max'])
        self.input_dim, self.sol_dim = params['input_dim'], params['sol_dim']

    def hamiltonian(self, J, u, points):
        t, x = points[:, 0:1], points[:, 1:]
        Jx = div(J, points, i=1)
        Jxx = laplacian(J, points, start_dim=1)

        drift_term = ((self.μ - self.r) * u + self.r) * x * Jx
        diffusion_term = 0.5 * (self.σ * u * x) ** 2 * Jxx
        return drift_term + diffusion_term

    def terminal_cost(self, points):
        t, x = points[:, 0:1], points[:, 1:]
        return x ** self.γ


class HJBSolver:
    def __init__(self, model_config, hjb_config, actor_net, critic_net):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("Using CPU!")
            self.device = torch.device("cpu")

        self.hjb = hjb_config
        self.config = model_config

        self.actor_net = actor_net.to(self.device)
        self.critic_net = critic_net.to(self.device)

        self._initialize_optimizers()
        self._initialize_samplers()
        self._initialize_loss_functions()
        self._initialize_alpha_annealing()

        self.gpinn_weight = self.config.get("gpinn_weight", 0.1)

    def _initialize_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.config["lr_actor"])
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.config["lr_critic"])
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=self.config["lr_decay_step"],
                                      gamma=self.config["lr_decay_gamma"])
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=self.config["lr_decay_step"],
                                       gamma=self.config["lr_decay_gamma"])

    def _initialize_samplers(self):
        self.domain_sampler = ControlledDriftSamplerHJB(
            hjb_config=self.hjb,
            T=self.hjb.T,
            x_domain=self.hjb.x_domain,
            batch_size=self.config["domain_batch_size"],
            device=self.device,
            n_range=self.config["sde_n_range"],
            boundary_memory_size=self.config["boundary_memory_size"]
        )
        self.terminal_sampler = TerminalSampler(
            domain_sampler=self.domain_sampler,
            batch_size=self.config["terminal_batch_size"]
        )
    def _initialize_loss_functions(self):
        self.terminal_criterion = lambda J, points: \
            torch.mean(torch.square(J - self.hjb.terminal_cost(points)))

        self.hamiltonian_criterion = lambda J, u, points: \
            torch.mean(-self.hjb.hamiltonian(J, u, points))

    def _initialize_alpha_annealing(self):
        self.initial_alpha = self.config.get("initial_alpha", 0.0)
        self.final_alpha = self.config.get("final_alpha", 1.0)
        self.alpha_anneal_end = self.config.get("alpha_anneal_end", 5000)
        self.current_alpha = self.initial_alpha

    def _update_alpha(self, iteration):
        if iteration < self.alpha_anneal_end:
            self.current_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * (
                        iteration / self.alpha_anneal_end)
        else:
            self.current_alpha = self.final_alpha

    def _compute_critic_loss(self, domain_points, terminal_points):
        domain_points.requires_grad_(True)

        with torch.no_grad():
            u_domain = self.actor_net(domain_points)
            u_effective = self.current_alpha * u_domain

        J_domain = self.critic_net(domain_points)
        J_terminal = self.critic_net(terminal_points)

        hjb_residual = div(J_domain, domain_points, i=0) + self.hjb.hamiltonian(J_domain, u_effective, domain_points)

        loss_f = torch.mean(hjb_residual ** 2)

        residual_t = div(hjb_residual, domain_points, i=0)
        residual_x = div(hjb_residual, domain_points, i=1)
        loss_g = torch.mean(residual_t ** 2) + torch.mean(residual_x ** 2)

        loss_terminal = self.terminal_criterion(J_terminal, terminal_points)

        total_critic_loss = (loss_f + self.gpinn_weight * loss_g) + self.config["terminal_weight"] * loss_terminal

        return total_critic_loss, {
            "hjb_f": loss_f.item(),
            "hjb_g": loss_g.item(),
            "terminal": loss_terminal.item()
        }

    def _compute_actor_loss(self, domain_points):

        domain_points_actor = domain_points.detach().requires_grad_(True)
        u_actor = self.actor_net(domain_points_actor)
        J_actor = self.critic_net(domain_points_actor)
        return self.hamiltonian_criterion(J_actor, u_actor, domain_points_actor)

    def solve(self, num_iterations: int):
        loss_history = []
        pbar = tqdm(range(num_iterations), desc="Solving HJB (gPINN)", ncols=120)

        for i in pbar:
            self.actor_net.train()
            self.critic_net.train()

            self._update_alpha(i)

            domain_points = self.domain_sampler.sample_batch()
            terminal_points = self.terminal_sampler.sample_batch()

            self.critic_optimizer.zero_grad()
            critic_loss, critic_loss_components = self._compute_critic_loss(domain_points, terminal_points)
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_scheduler.step()

            actor_loss_val = 0.0
            if (i + 1) % self.config.get("delay_actor", 1) == 0:
                self.actor_optimizer.zero_grad()
                actor_loss = self._compute_actor_loss(domain_points)
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_scheduler.step()
                actor_loss_val = actor_loss.item()

            with torch.no_grad():
                u_for_sampler = self.actor_net(domain_points)
            u_effective_sampler = self.current_alpha * u_for_sampler
            self.domain_sampler.update(u_effective_sampler)

            loss_dict = {
                'critic_total': critic_loss.item(),
                'critic_hjb_f': critic_loss_components['hjb_f'],
                'critic_hjb_g': critic_loss_components['hjb_g'],
                'critic_terminal': critic_loss_components['terminal'],
                'actor_hamiltonian': actor_loss_val
            }
            loss_history.append(loss_dict)

            if (i + 1) % 100 == 0:
                postfix_dict = OrderedDict()
                postfix_dict['critic'] = f"{loss_dict['critic_total']:.3e}"
                postfix_dict['actor'] = f"{loss_dict['actor_hamiltonian']:.3e}"
                postfix_dict['alpha'] = f"{self.current_alpha:.2f}"
                pbar.set_postfix(ordered_dict=postfix_dict)

        print("Training finished.")

        return loss_history
