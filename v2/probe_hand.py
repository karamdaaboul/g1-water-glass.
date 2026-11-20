"""Probe (2026-11-18) of the right Dex3 hand on the fixed-base G1 (robot only): finger joint limits, and the finger link positions
in the palm frame with the hand open (0) and with the fingers at a few closing levels (fraction of each joint's range towards the
side that flexes the finger)."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_glass_env import G1GlassEnvCfg  # noqa: E402
cfg = G1GlassEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import math as mu
    env = ManagerBasedRLEnv(cfg=cfg); env.reset(); r = env.scene["robot"]
    fj = [n for n in r.joint_names if n.startswith("right_hand_")]
    ids = [r.joint_names.index(n) for n in fj]
    lim = r.data.soft_joint_pos_limits.torch[0, ids]
    print("FINGER JOINTS", {n: [round(float(a), 3), round(float(b), 3)] for n, (a, b) in zip(fj, lim.tolist())}, flush=True)
    links = [n for n in r.body_names if n.startswith("right_hand_")]
    print("HAND LINKS", links, flush=True)
    palm = r.body_names.index("right_hand_palm_link")
    zero = torch.zeros(1, 7, device=env.device)
    for frac in (0.0, 0.3, 0.6):
        q = r.data.joint_pos.torch.clone()
        for j, (a, b) in zip(ids, lim.tolist()):
            q[0, j] = frac * (b if abs(b) > abs(a) else a)  # towards the larger end of the range
        r.write_joint_position_to_sim_index(position=q); r.set_joint_position_target_index(target=q)
        env.action_manager.get_term("right_arm").reset()
        for _ in range(3):
            env.step(zero)
        pp, pq = r.data.body_pos_w.torch[0, palm], r.data.body_quat_w.torch[0, palm]
        rel = {n: [round(x, 3) for x in mu.quat_apply_inverse(pq[None], (r.data.body_pos_w.torch[0, r.body_names.index(n)] - pp)[None])[0].tolist()] for n in links}
        print(f"FRAC {frac}: finger q {[round(float(r.data.joint_pos.torch[0, j]), 2) for j in ids]}", flush=True)
        print(f"   links in palm frame: {rel}", flush=True)
    env.close()
