import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from DGMNet import DGMNet, ResNet
from ControlledDriftSamplerHJB import ControlledDriftSamplerHJB, TerminalSampler
from HJBSolver import HJBSolver, RiskyAssetConfig, div, laplacian
from collections import OrderedDict

if __name__ == '__main__':

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")


    HJB_PARAMS = {
        'mu': 0.06,
        'sigma': 0.6,
        'r': 0.03,
        'gamma': 0.7,
        'T': 1.0,
        'x_min': 0.1,
        'x_max': 10.0,
        'input_dim': 2,
        'sol_dim': 1
    }

    TRAIN_PARAMS = {
        'num_iterations': 10000,
        'domain_batch_size': 4096,
        'terminal_batch_size': 2048,
        'boundary_memory_size': 4096,
        'lr_actor': 1e-4,
        'lr_critic': 1e-4,
        'lr_decay_step': 100,
        'lr_decay_gamma': 0.99,
        'terminal_weight': 2.0,
        'delay_actor': 5,
        'initial_alpha': 0.0,
        'final_alpha': 1.0,
        'alpha_anneal_end': 8000,
        'sde_n_range': (20, 40),
        'gpinn_weight': 0.1,
    }

    NET_PARAMS = {
        'actor_hidden_dim': 24,
        'actor_num_layers': 3,
        'critic_hidden_dim': 64,
        'critic_num_layers': 6,
    }

    hjb_config_instance = RiskyAssetConfig(HJB_PARAMS)

    lb = torch.tensor([0.0, HJB_PARAMS['x_min']], dtype=torch.float32).to(DEVICE)
    ub = torch.tensor([HJB_PARAMS['T'], HJB_PARAMS['x_max']], dtype=torch.float32).to(DEVICE)

    actor_net = ResNet(input_dim=HJB_PARAMS['input_dim'],
                       hidden_dim=NET_PARAMS['actor_hidden_dim'],
                       num_layers=NET_PARAMS['actor_num_layers'],
                       output_dim=HJB_PARAMS['sol_dim'],
                       lb=lb, ub=ub,
                       constrain_output=True).to(DEVICE)

    critic_net = DGMNet(input_dim=HJB_PARAMS['input_dim'],
                        hidden_dim=NET_PARAMS['critic_hidden_dim'],
                        num_layers=NET_PARAMS['critic_num_layers'],
                        lb=lb, ub=ub).to(DEVICE)

    solver = HJBSolver(
        model_config=TRAIN_PARAMS,
        hjb_config=hjb_config_instance,
        actor_net=actor_net,
        critic_net=critic_net
    )

    losses = solver.solve(num_iterations=TRAIN_PARAMS['num_iterations'])

    ACTOR_SAVE_PATH = 'hjb_actor_model.pth'
    CRITIC_SAVE_PATH = 'hjb_critic_model.pth'

    torch.save(solver.actor_net.state_dict(), ACTOR_SAVE_PATH)
    torch.save(solver.critic_net.state_dict(), CRITIC_SAVE_PATH)

    print("\n" + "=" * 50)
    print("           Training Complete!")
    print(f"   Actor model saved to: {ACTOR_SAVE_PATH}")
    print(f"  Critic model saved to: {CRITIC_SAVE_PATH}")
    print("=" * 50)

