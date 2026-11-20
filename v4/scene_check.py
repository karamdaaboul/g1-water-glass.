"""Scene check (2026-11-19): glass stands on the table (weld off); weld on -> lift the arm (shoulder pitch) -> the glass follows;
weld off -> the glass drops back. Prints glass motion, palm-glass distance and spill per phase."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_pp_env import G1PPEnv, G1PPEnvCfg  # noqa: E402
cfg = G1PPEnvCfg(); cfg.scene.num_envs = 2; cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import torch
    env = G1PPEnv(cfg=cfg); env.reset()
    n, d = env.num_envs, env.device
    print(f"WELDS {env._weld_idx.tolist()}", flush=True)
    z = torch.zeros(n, 7, device=d)
    g0 = env.glass_pose_w()[0].clone()
    for _ in range(15):
        env.step(z)
    g1 = env.glass_pose_w()[0]
    print(f"STAND glass moved {float((g1 - g0).norm(dim=-1).max()) * 1000:.1f} mm, tilt {float(env.glass_tilt().max()) * 57.3:.2f} deg, spill {float(env.spilled_fraction().max()):.4f}", flush=True)
    ids = torch.arange(n, device=d)
    env.set_hand(ids, True); env.set_weld(ids, True)
    d0 = (env.glass_pose_w()[0] - env.palm_pose_w()[0]).norm(dim=-1).clone()
    a = torch.zeros(n, 7, device=d); a[:, 3] = -0.3  # elbow: raise the forearm
    for _ in range(10):
        env.step(a)
    g2 = env.glass_pose_w()[0]; d1 = (g2 - env.palm_pose_w()[0]).norm(dim=-1)
    print(f"LIFT glass rose {float((g2 - g1)[:, 2].min()) * 1000:.1f} mm, palm-glass distance change {float((d1 - d0).abs().max()) * 1000:.2f} mm, tilt {float(env.glass_tilt().max()) * 57.3:.2f} deg, spill {float(env.spilled_fraction().max()):.4f}", flush=True)
    env.set_weld(ids, False); env.set_hand(ids, False)
    for _ in range(15):
        env.step(z)
    g3 = env.glass_pose_w()[0]
    print(f"DROP glass z {float(g3[:, 2].min() - env.scene.env_origins[:, 2].min()):.3f} m (standing height {0.75 + 0.006 + 0.09:.3f}), tilt {float(env.glass_tilt().max()) * 57.3:.1f} deg, spill {float(env.spilled_fraction().max()):.4f}", flush=True)
    env.close()
