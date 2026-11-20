"""Debug (2026-11-19): the normal glass's water right after reset and while the glass stands still."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_pp_env import BOTTOM, H_GLASS, R_IN, G1PPEnv, G1PPEnvCfg  # noqa: E402
cfg = G1PPEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu
    env = G1PPEnv(cfg=cfg); env.reset()
    w = env.water_sim
    print(f"PBF particles {w.P}, presettle level error {getattr(w, 'presettle_level_error', None)}", flush=True)
    def show(tag):
        gp, gq = env.glass_pose_w(); x = w.positions()[0]
        loc = mu.quat_apply_inverse(gq[0][None].expand(len(x), -1), x - gp[0])
        r = loc[:, :2].norm(dim=-1)
        print(f"{tag}: spill {float(env.spilled_fraction()[0]):.3f}; local z {float(loc[:, 2].min()) * 100:.1f}..{float(loc[:, 2].max()) * 100:.1f} cm (rim {H_GLASS * 100:.1f}); "
              f"r max {float(r.max()) * 100:.2f} cm (R_IN {R_IN * 100:.1f}); glass z {float(gp[0, 2]):.3f}", flush=True)
    show("reset")
    for k in range(10):
        env.step(torch.zeros(1, 7, device=env.device))
        if k in (0, 4, 9):
            show(f"step {k}")
    env.close()
