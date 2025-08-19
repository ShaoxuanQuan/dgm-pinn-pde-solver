import torch

def compute_residual_and_grads(points: torch.Tensor, model: torch.nn.Module, pde_params: dict):

    if not points.requires_grad:
        points.requires_grad_(True)

    V = model(points)
    S = points[:, 1:2]
    r, sigma = pde_params['r'], pde_params['sigma']

    grad_V = torch.autograd.grad(V, points, torch.ones_like(V), create_graph=True)[0]
    V_t, V_s = grad_V[:, 0:1], grad_V[:, 1:2]

    grad_V_s = torch.autograd.grad(V_s, points, torch.ones_like(V_s), create_graph=True)[0]
    V_ss = grad_V_s[:, 1:2]

    residual = V_t + r * S * V_s + 0.5 * (sigma ** 2) * (S ** 2) * V_ss - r * V

    # 计算残差 R 的梯度 (dR/dt, dR/dS)
    grad_residual = torch.autograd.grad(residual, points, torch.ones_like(residual), create_graph=True)[0]
    residual_t, residual_s = grad_residual[:, 0:1], grad_residual[:, 1:2]

    return {
        'residual': residual,
        'residual_t': residual_t,
        'residual_s': residual_s
    }


'''======================================================================================='''

def compute_residual_and_grads_2D(points: torch.Tensor, model: torch.nn.Module, pde_params: dict):

    if not points.requires_grad:
        points.requires_grad_(True)

    V = model(points)
    S1 = points[:, 1:2]
    S2 = points[:, 2:3]

    r = pde_params['r']
    sigma1 = pde_params['sigma1']
    sigma2 = pde_params['sigma2']
    rho = pde_params['rho']

    grad_V = torch.autograd.grad(V, points, torch.ones_like(V), create_graph=True)[0]
    V_t = grad_V[:, 0:1]
    V_s1 = grad_V[:, 1:2]
    V_s2 = grad_V[:, 2:3]

    grad_V_s1 = torch.autograd.grad(V_s1, points, torch.ones_like(V_s1), create_graph=True)[0]
    V_s1s1 = grad_V_s1[:, 1:2]

    grad_V_s2 = torch.autograd.grad(V_s2, points, torch.ones_like(V_s2), create_graph=True)[0]
    V_s2s2 = grad_V_s2[:, 2:3]

    V_s1s2 = grad_V_s1[:, 2:3]

    residual = (
            V_t
            + r * S1 * V_s1
            + r * S2 * V_s2
            + 0.5 * (sigma1 ** 2) * (S1 ** 2) * V_s1s1
            + 0.5 * (sigma2 ** 2) * (S2 ** 2) * V_s2s2
            + rho * sigma1 * sigma2 * S1 * S2 * V_s1s2
            - r * V
    )

    grad_residual = torch.autograd.grad(residual, points, torch.ones_like(residual), create_graph=True)[0]
    residual_t = grad_residual[:, 0:1]
    residual_s1 = grad_residual[:, 1:2]
    residual_s2 = grad_residual[:, 2:3]

    return {
        'residual': residual,
        'residual_t': residual_t,
        'residual_s1': residual_s1,
        'residual_s2': residual_s2,
    }
