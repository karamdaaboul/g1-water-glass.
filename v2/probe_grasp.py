"""Grasp probe (2026-11-18): the glass starts weightless (G1H_GLASS_NOGRAV=1) at the planned grip pose with the hand open; the
fingers close over 2 s (targets open -> closed), then hold 1 s. Prints per 5 steps the finger joints and the glass pose in the
palm frame; at the end, the contact pose of the fingers and the glass offset to use (GRIP, HAND_GRASP)."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_hold_env import H_GLASS, HAND_CLOSED, HAND_OPEN, G1HoldEnv, G1HoldEnvCfg, set_mpm_capacity  # noqa: E402
cfg = G1HoldEnvCfg(); cfg.scene.num_envs = 2; cfg.sim.device = args.device; set_mpm_capacity(cfg)
with launch_simulation(cfg, args):
    import torch
    env = G1HoldEnv(cfg=cfg); env.reset()
    r = env.robot; h = env.hand_ids; zero = torch.zeros(env.num_envs, 7, device=env.device)
    o, c = torch.tensor(HAND_OPEN, device=env.device), torch.tensor(HAND_CLOSED, device=env.device)
    print("START glass in palm", [round(x, 4) for x in env.glass_in_palm()[0][0].tolist()], flush=True)
    for k in range(45):
        a = min(1.0, k / 30)
        r.set_joint_position_target_index(target=(o + a * (c - o)).expand(env.num_envs, -1), joint_ids=h)
        env.step(zero)
        if k % 5 == 4 or k == 44:
            rel, ang = env.glass_in_palm()
            print(f"k {k:2d} fingers {[round(x, 3) for x in r.data.joint_pos.torch[0, h].tolist()]} glass in palm {[round(x, 4) for x in rel[0].tolist()]} axis-palm z {ang[0] * 57.3:.1f} deg", flush=True)
    rel, ang = env.glass_in_palm()
    q = r.data.joint_pos.torch[0, h]
    print(f"RESULT HAND_GRASP={','.join(f'{x:.4f}' for x in q.tolist())}  GRIP={rel[0, 0]:.4f},{rel[0, 1]:.4f},{rel[0, 2] + 0.5 * H_GLASS:.4f}  axis {ang[0] * 57.3:.1f} deg", flush=True)
    env.close()
