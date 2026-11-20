"""Throughput (2026-11-19) of the v3 wine-glass env: 10 warm-up steps, then 40 timed steps of small random commands."""
import argparse, time
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--num_envs", type=int, default=16); add_launcher_args(parser); args = parser.parse_args()
from g1_wine_env import G1InHandEnv, G1InHandEnvCfg, set_mpm_capacity  # noqa: E402
cfg = G1InHandEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device; set_mpm_capacity(cfg)
with launch_simulation(cfg, args):
    import torch
    env = G1InHandEnv(cfg=cfg); env.reset()
    for _ in range(10):
        env.step(0.2 * torch.randn(env.num_envs, 7, device=env.device))
    torch.cuda.synchronize(); t = time.time()
    for _ in range(40):
        env.step(0.2 * torch.randn(env.num_envs, 7, device=env.device))
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"SPEED {args.num_envs} envs: {40 * args.num_envs / dt:.0f} env-steps/s ({dt / 40 * 1000:.0f} ms per policy step)", flush=True)
    env.close()
