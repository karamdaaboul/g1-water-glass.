"""Why does the water move in a glass nobody touches? (2026-11-19, user question)

A) the water alone, glass pose held exactly fixed (no sim, no robot): particle speed over 4 s after a reset;
B) the pick-and-place env, glass standing on the table, arm commands zero (as before a grasp): glass pose jitter from MuJoCo
   (finite differences of its pose) and the particle speed over the same 4 s.
If A moves as much as B, the motion is the solver's own; if only B moves, the glass jitter drives it.
"""
import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=4.0)
add_launcher_args(parser)
args = parser.parse_args()

from g1_pp_env import BOTTOM, CAVITY, FREE, H_GLASS, R_IN, R_OUT  # noqa: E402
from g1_pp_task import G1PPTaskEnv, G1PPTaskEnvCfg  # noqa: E402

cfg = G1PPTaskEnvCfg()
cfg.scene.num_envs = 4
cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import numpy as np
    import torch
    import warp as wp

    from pbf_water import PBFWater

    dt = 1.0 / 15.0
    steps = int(args.seconds / dt)

    def speed_stats(w):
        v = wp.to_torch(w.v).view(w.E, w.P, 3)
        act = wp.to_torch(w.active).view(w.E, w.P).bool()
        s = v.norm(dim=-1)[act]
        return float(s.pow(2).mean().sqrt()), float(s.max())

    # A) water alone, glass exactly still
    d = args.device
    w = PBFWater(4, R_IN, R_OUT, BOTTOM, H_GLASS, CAVITY - FREE, d, spacing=0.005, substeps=16, iterations=6)
    w.presettle(CAVITY - FREE)
    pos = torch.zeros(4, 3, device=d); pos[:, 0] = torch.arange(4, device=d) * 2.0
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=d).repeat(4, 1)
    w.reset(torch.arange(4, device=d), pos, quat)
    rowsA = []
    for k in range(steps):
        w.step(pos, quat, pos, quat, dt)
        rowsA.append(speed_stats(w))
    # B) the env: glass on the table, arm still
    env = G1PPTaskEnv(cfg=cfg)
    env.reset()
    rowsB, gv = [], []
    for k in range(steps):
        p0 = env.glass_pose_w()[0].clone()
        q0 = env.glass_pose_w()[1].clone()
        env.step(torch.zeros(4, 8, device=d) - torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.0], device=d))  # arm still, hand open
        p1, q1 = env.glass_pose_w()
        dq = (q1 * torch.cat([-q0[:, :3], q0[:, 3:]], -1)).sum(-1).abs().clamp(max=1.0)
        gv.append((float(((p1 - p0) / dt).norm(dim=-1).max()), float((2 * torch.acos(dq) / dt).max())))
        rowsB.append(speed_stats(env.water_sim))
    for name, rows in (("A water alone, glass fixed", rowsA), ("B env, glass on the table", rowsB)):
        r = np.array(rows)
        print(f"{name}: particle speed rms (m/s) at 0.5 s {r[7, 0]:.4f}, 1 s {r[14, 0]:.4f}, 2 s {r[29, 0]:.4f}, end {r[-1, 0]:.4f}; "
              f"max particle speed at the end {r[-1, 1]:.3f}", flush=True)
    g = np.array(gv)
    print(f"B glass jitter: linear speed max {g[:, 0].max() * 1000:.2f} mm/s (median {np.median(g[:, 0]) * 1000:.2f}), "
          f"angular {np.degrees(g[:, 1].max()):.3f} deg/s", flush=True)
    env.close()
