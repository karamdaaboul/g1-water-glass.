"""Throughput (2026-11-18) of the stage-2 water env: 20 warm-up steps, then 60 timed steps of small random commands."""
import argparse
import time
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--num_envs", type=int, default=64); add_launcher_args(parser); args = parser.parse_args()
from g1_water_env import G1WaterEnv, G1WaterEnvCfg, set_mpm_capacity  # noqa: E402
cfg = G1WaterEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device; set_mpm_capacity(cfg)
with launch_simulation(cfg, args):
    import torch
    env = G1WaterEnv(cfg=cfg); env.reset()
    for _ in range(20):
        env.step(0.3 * torch.randn(env.num_envs, 7, device=env.device))
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(60):
        env.step(0.3 * torch.randn(env.num_envs, 7, device=env.device))
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"SPEED {args.num_envs} envs: {60 * args.num_envs / dt:.0f} env-steps/s ({dt / 60 * 1000:.0f} ms per policy step), spilled mean {env.spilled_fraction().mean().item():.3f}, GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    env.close()
