import numpy as np
import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pilco.rewards import ExponentialReward
from pilco.controllers import LinearController
from pilco.models import PILCO


class CartPoleEnv:
    """Custom CartPole environment matching pilco_python dynamics_cp.py physics."""

    def __init__(self, dt=0.10):
        self.dt = dt
        self.l = 0.5
        self.m = 0.5
        self.M = 0.5
        self.b = 0.1
        self.g = 9.82
        self.maxU = 10.0
        self.state = None

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        noise = 0.01 * np.random.randn(4)
        self.state += noise
        return self.state.copy()

    def step(self, action):
        u = np.clip(np.asarray(action).ravel()[0], -self.maxU, self.maxU)
        z = self.state.copy()
        x, v, dtheta, theta = z

        dz = np.zeros(4)
        dz[0] = v
        dz[1] = (2 * self.m * self.l * dtheta**2 * np.sin(theta)
                 + 3 * self.m * self.g * np.sin(theta) * np.cos(theta)
                 + 4 * u - 4 * self.b * v) / (4 * (self.M + self.m)
                 - 3 * self.m * np.cos(theta)**2)
        dz[2] = (-3 * self.m * self.l * dtheta**2 * np.sin(theta) * np.cos(theta)
                 - 6 * (self.M + self.m) * self.g * np.sin(theta)
                 - 6 * (u - self.b * v) * np.cos(theta)) / (4 * self.l * (self.m + self.M)
                 - 3 * self.m * self.l * np.cos(theta)**2)
        dz[3] = dtheta

        self.state = z + self.dt * dz
        return self.state.copy()


def rollout(env, pilco, timesteps, random=False):
    X = []
    Y = []
    x = env.reset()
    for _ in range(timesteps):
        if random:
            u = np.random.uniform(-env.maxU, env.maxU)
        else:
            u = pilco.compute_action(x[None, :])[0, :]
            u = u.detach().cpu().numpy()
        x_new = env.step(u)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)


def run_cartpole_gpytorch(num_iterations=3, horizon=40, output_dir="."):
    print("=" * 60)
    print("  PILCO-gpytorch CartPole Swingup Benchmark")
    print("=" * 60)

    env = CartPoleEnv(dt=0.10)
    state_dim = 4
    control_dim = 1

    # Phase 1: Initial random rollouts
    print("\n-- Phase 1: Initial random rollouts --")
    t0 = time.time()
    X, Y = rollout(env, pilco=None, timesteps=horizon, random=True)
    for _ in range(0):
        X_, Y_ = rollout(env, pilco=None, timesteps=horizon, random=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))
    print(f"  Random rollout: {len(X)} steps in {time.time() - t0:.1f}s")

    # Controller: Linear controller with state_dim=4, control_dim=1
    controller = LinearController(state_dim=state_dim, control_dim=control_dim,
                                 max_action=torch.tensor([10.0]).float().cuda())

    # Reward: exponential reward, target = [0, 0, 0, pi]
    R = ExponentialReward(state_dim=state_dim,
                          W=0.5 * np.diag([1.0, 0.1, 0.1, 1.0]),
                          t=np.array([0.0, 0.0, 0.0, np.pi]))

    m_init = np.reshape(np.array([0.0, 0.0, 0.0, 0.0]), (1, state_dim))
    S_init = np.diag([0.01, 0.01, 0.01, 0.01])
    m_init = torch.from_numpy(m_init).float().cuda()
    S_init = torch.from_numpy(S_init).float().cuda()

    print("\n-- Phase 2: Train GP dynamics model --")
    pilco = PILCO(X, Y, controller=controller, horizon=horizon,
                  reward=R, m_init=m_init, S_init=S_init)
    t0 = time.time()
    pilco.optimize_models(maxiter=300, restarts=1)
    print(f"  GP training done in {time.time() - t0:.1f}s")

    # Phase 3-N: PILCO iterations
    print(f"\n-- Phase 3: PILCO iterations ({num_iterations} total) --")
    total_gp_time = 0.0
    total_policy_time = 0.0

    for j in range(num_iterations):
        print(f"\n  >>> Iteration {j + 1}/{num_iterations} <<<")
        t_start = time.time()

        # Optimize policy
        t_pol = time.time()
        pilco.optimize_policy(maxiter=50, restarts=1)
        policy_time = time.time() - t_pol
        total_policy_time += policy_time

        # Controlled rollout
        X_new, Y_new = rollout(env, pilco=pilco, timesteps=horizon * 2, random=False)
        theta_range = [X_new[:, 3].min(), X_new[:, 3].max()]
        near_upright = np.any(np.abs(X_new[:, 3] - np.pi) < 0.5)
        print(f"  Rollout: theta=[{theta_range[0]:.1f}, {theta_range[1]:.1f}] "
              f"upright={near_upright} samples={len(X_new)}")

        # Update and re-train GP
        X = np.vstack((X, X_new))
        Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)

        t_gp = time.time()
        pilco.optimize_models(maxiter=100, restarts=1)
        gp_time = time.time() - t_gp
        total_gp_time += gp_time

        print(f"  Time: {time.time() - t_start:.1f}s | "
              f"policy={policy_time:.1f}s gp={gp_time:.1f}s "
              f"total_data={len(X)}")

        if near_upright:
            print(f"\n  *** SWING-UP ACHIEVED at iteration {j + 1}! ***")
            break

    # Final evaluation
    print(f"\n-- Summary --")
    final_reward = pilco.compute_reward()
    print(f"  Final policy reward: {final_reward.item():.4f}")
    print(f"  Total GP time: {total_gp_time:.1f}s")
    print(f"  Total policy time: {total_policy_time:.1f}s")
    print(f"  Total data: {len(X)} steps")

    return {
        'final_reward': final_reward.item(),
        'total_gp_time': total_gp_time,
        'total_policy_time': total_policy_time,
        'total_data': len(X),
        'pilco': pilco,
        'env': env,
    }


if __name__ == '__main__':
    results = run_cartpole_gpytorch(num_iterations=3, horizon=40)
