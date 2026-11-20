"""Placement probe (2026-11-18): fingers in Isaac Lab's Dex3 gripper pose; the weightless glass (axis = palm z, mid-height at the
palm plane) at one candidate (x, y) of a 9 x 9 grid per env (palm frame, 5 mm steps); 4 steps; displacement of the glass = how
hard the hand pushed it out (0 = no overlap). Prints the free placements and the one farthest from any pushed placement."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_hold_env import H_GLASS, RIGHT_HAND, G1HoldEnv, G1HoldEnvCfg, set_mpm_capacity  # noqa: E402
XS = [0.045 + 0.005 * i for i in range(9)]; YS = [0.010 + 0.005 * j for j in range(9)]
cfg = G1HoldEnvCfg(); cfg.scene.num_envs = len(XS) * len(YS); cfg.sim.device = args.device; set_mpm_capacity(cfg)
GRIP_POSE = (1.0, 1.0, 1.0, 1.0, 0.0, -0.4, -0.7)
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu
    env = G1HoldEnv(cfg=cfg); env.reset()
    r, h, n, d = env.robot, env.hand_ids, env.num_envs, env.device
    g = torch.tensor(GRIP_POSE, device=d).expand(n, -1)
    q = r.data.joint_pos.torch.clone(); q[:, h] = g
    r.write_joint_position_to_sim_index(position=q); r.set_joint_position_target_index(target=g, joint_ids=h)
    env.scene.write_data_to_sim(); env.sim.forward(); r.update(0.0)
    cand = torch.tensor([(x, y) for x in XS for y in YS], device=d)
    off = torch.cat([cand, torch.full((n, 1), -0.5 * H_GLASS, device=d)], -1)
    pp, pq = r.data.body_pos_w.torch[:, env.palm], r.data.body_quat_w.torch[:, env.palm]
    start = pp + mu.quat_apply(pq, off)
    env.glass.write_root_pose_to_sim_index(root_pose=torch.cat([start, pq], -1), env_ids=torch.arange(n, device=d))
    env.glass.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=torch.arange(n, device=d))
    env.scene.write_data_to_sim(); env.sim.forward()
    for _ in range(4):
        env.step(torch.zeros(n, 7, device=d))
    rel, _ = env.glass_in_palm()
    disp = (rel[:, :2] - cand).norm(dim=-1) * 1000
    free = disp < 1.0
    print("DISPLACEMENT mm (rows x = " + ", ".join(f"{x:.3f}" for x in XS) + "; cols y = " + ", ".join(f"{y:.3f}" for y in YS) + ")", flush=True)
    for i in range(len(XS)):
        print("   " + " ".join(f"{float(disp[i * len(YS) + j]):6.1f}" for j in range(len(YS))), flush=True)
    if free.any():
        pushed = cand[~free]
        clear = torch.cdist(cand[free], pushed).min(-1).values if len(pushed) else torch.ones(int(free.sum()), device=d)
        k = torch.argmax(clear)
        print(f"BEST free placement x {float(cand[free][k, 0]):.3f} y {float(cand[free][k, 1]):.3f} (clearance {float(clear[k]) * 1000:.1f} mm to a pushed one)", flush=True)
    env.close()
