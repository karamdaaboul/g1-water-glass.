"""Water level probe (2026-11-19): hold still 15 steps, then the particle heights in the glass frame (median of the top layer =
surface) vs the rim, and the radial extent. --module g1_wine_env (v3) or g1_inhand_env (v2, with G1I_WATER_H)."""
import argparse, importlib
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--module", default="g1_wine_env"); add_launcher_args(parser); args = parser.parse_args()
M = importlib.import_module(args.module)
cfg = M.G1InHandEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device; M.set_mpm_capacity(cfg)
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu
    env = M.G1InHandEnv(cfg=cfg); env.reset()
    for _ in range(15):
        env.step(torch.zeros(1, 7, device=env.device))
    pos, q = env.cup_pose_w(); pts = env.water.data.particle_pos_w.torch[0]
    loc = mu.quat_apply_inverse(q[0][None].expand(pts.shape[0], -1), pts - pos[0])
    z = loc[:, 2].sort().values; r = loc[:, :2].norm(dim=-1)
    top = z[-max(1, len(z) // 20):].median()  # top 5 %
    print(f"LEVEL {args.module}: particles {len(z)}, water top (median of top 5 %) {top * 100:.2f} cm, max {z[-1] * 100:.2f} cm, rim {M.H_GLASS * 100:.2f} cm "
          f"-> free {(M.H_GLASS - top) * 100:.2f} cm; inner radius {M.R_IN * 100:.2f} cm, particle r max {r.max() * 100:.2f} cm; spill {env.spilled_fraction()[0]:.4f}", flush=True)
    env.close()
