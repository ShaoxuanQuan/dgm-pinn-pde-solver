import torch
import numpy as np
from DGMNet import DGMNet, ResNet


def evaluate_hjb_solution():

    print("--- 开始配置评估环境 ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")

    HJB_PARAMS = {
        'mu': 0.06, 'sigma': 0.6, 'r': 0.03, 'gamma': 0.7,
        'T': 1.0, 'x_min': 0.1, 'x_max': 10.0,
        'input_dim': 2, 'sol_dim': 1
    }

    NET_PARAMS = {
        'actor_hidden_dim': 24, 'actor_num_layers': 3,
        'critic_hidden_dim': 64, 'critic_num_layers': 6,
    }

    ACTOR_SAVE_PATH = 'hjb_actor_model.pth'
    CRITIC_SAVE_PATH = 'hjb_critic_model.pth'

    print("--- 正在加载模型权重 ---")

    lb = torch.tensor([0.0, HJB_PARAMS['x_min']], dtype=torch.float32).to(DEVICE)
    ub = torch.tensor([HJB_PARAMS['T'], HJB_PARAMS['x_max']], dtype=torch.float32).to(DEVICE)

    actor_model = ResNet(
        input_dim=HJB_PARAMS['input_dim'], hidden_dim=NET_PARAMS['actor_hidden_dim'],
        num_layers=NET_PARAMS['actor_num_layers'], output_dim=HJB_PARAMS['sol_dim'],
        lb=lb, ub=ub, constrain_output=True
    ).to(DEVICE)

    critic_model = DGMNet(
        input_dim=HJB_PARAMS['input_dim'], hidden_dim=NET_PARAMS['critic_hidden_dim'],
        num_layers=NET_PARAMS['critic_num_layers'], lb=lb, ub=ub
    ).to(DEVICE)

    try:
        actor_model.load_state_dict(torch.load(ACTOR_SAVE_PATH, map_location=DEVICE))
        critic_model.load_state_dict(torch.load(CRITIC_SAVE_PATH, map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"错误: 找不到模型文件 {e.filename}。请确保脚本与.pth文件在同一目录下。")
        return

    actor_model.eval()
    critic_model.eval()
    print("模型加载成功！")


    print("\n--- 开始进行定量评估 ---")

    params = HJB_PARAMS
    u_sol_func = lambda t: (params['mu'] - params['r']) / (params['sigma'] ** 2 * (1 - params['gamma']))
    J_sol_func = lambda x, t: x ** params['gamma'] * np.exp(
        (params['gamma'] * (params['mu'] - params['r']) ** 2) /
        (params['sigma'] ** 2 * (2 * params['gamma'] - 2)) * (params['T'] - t)
    )

    num_points_t = 200
    num_points_x = 200
    t_space = np.linspace(0, params['T'], num_points_t)
    x_space = np.linspace(params['x_min'], params['x_max'], num_points_x)
    T_grid, X_grid = np.meshgrid(t_space, x_space)

    points_flat = np.vstack([T_grid.ravel(), X_grid.ravel()]).T
    points_tensor = torch.tensor(points_flat, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_J_flat = critic_model(points_tensor).cpu().numpy().flatten()

        pred_u_flat = actor_model(points_tensor).cpu().numpy().flatten()

    true_J_flat = J_sol_func(X_grid.ravel(), T_grid.ravel())
    true_u_val = u_sol_func(0)

    l2_error_J = np.linalg.norm(pred_J_flat - true_J_flat) / np.linalg.norm(true_J_flat)

    model_u_mean = np.mean(pred_u_flat)
    model_u_std = np.std(pred_u_flat)

    print("\n" + "=" * 50)
    print("           评 估 结 果")
    print("=" * 50)
    print(f"价值函数 J 的相对 L2 误差: {l2_error_J:.6e}")
    print("\n--- 最优控制 u 对比 ---")
    print(f"  模型解 (均值):   {model_u_mean:.6f}")
    print(f"  模型解 (标准差): {model_u_std:.6e}  (此值越小，说明模型解越接近常数)")
    print(f"  解析解:          {true_u_val:.6f}")
    print("=" * 50)


if __name__ == '__main__':
    evaluate_hjb_solution()