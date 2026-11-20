"""Target bank (2026-11-18): right-arm configurations of the fixed-base G1 that hold the glass upright, with the glass base position.

Robot-only env (g1_glass_env stage 1, fast). For each batch: sample shoulder pitch/roll/yaw, elbow and wrist yaw around HOME; then
solve wrist roll and pitch by Gauss-Newton on the glass up-vector (x, y) with finite differences, using the glass pose = palm pose x
the weld of g1_water_env. Keep tilt < 3 deg and the glass base inside the workspace box (in front of the body, above the tray).
Writes target_bank.pt: q (N, 7), cup_pos (N, 3) env frame, tilt (N,)."""
import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--batches", type=int, default=16)
parser.add_argument("--out", default="target_bank.pt")
add_launcher_args(parser)
args = parser.parse_args()

from g1_glass_env import RIGHT_ARM, G1GlassEnvCfg  # noqa: E402
from g1_water_env import HOME, _weld_local_pose  # noqa: E402

SPREAD = (0.7, 0.4, 0.7, 0.6, 0.0, 0.0, 0.6)  # +- around HOME per joint for the sample (wrist roll/pitch are solved)
BOX_LO, BOX_HI = (0.18, -0.15, 0.58), (0.55, 0.45, 0.95)  # glass base workspace, env frame (home base: 0.288, 0.205, 0.684)

cfg = G1GlassEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import math as mu

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    robot = env.scene["robot"]
    term = env.action_manager.get_term("right_arm")
    ids = [robot.joint_names.index(n) for n in RIGHT_ARM]
    palm = robot.body_names.index("right_hand_palm_link")
    lim = robot.data.soft_joint_pos_limits.torch[0, ids]
    wpos, wq = _weld_local_pose()
    wpos = torch.tensor(wpos, device=env.device)[None]
    wq = torch.tensor(wq, device=env.device)[None]
    zero = torch.zeros(env.num_envs, 7, device=env.device)

    def cup_pose(q_arm):
        q = robot.data.joint_pos.torch.clone()
        q[:, ids] = q_arm
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        robot.set_joint_position_target_index(target=q)
        term.reset()
        env.step(zero)
        p, qp = robot.data.body_pos_w.torch[:, palm], robot.data.body_quat_w.torch[:, palm]
        pos = p + mu.quat_apply(qp, wpos.expand(env.num_envs, -1)) - env.scene.env_origins
        quat = mu.quat_mul(qp, wq.expand(env.num_envs, -1))
        up = mu.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1))
        return pos, up

    home = torch.tensor(HOME, device=env.device)
    keep_q, keep_pos, keep_tilt = [], [], []
    for b in range(args.batches):
        q = home + (2 * torch.rand(env.num_envs, 7, device=env.device) - 1) * torch.tensor(SPREAD, device=env.device)
        q = torch.minimum(torch.maximum(q, lim[:, 0]), lim[:, 1])
        for _ in range(10):  # Gauss-Newton on (up_x, up_y) over wrist roll (4) and pitch (5)
            _, up = cup_pose(q)
            r = up[:, :2]
            J = torch.zeros(env.num_envs, 2, 2, device=env.device)
            for c, j in enumerate((4, 5)):
                dq = q.clone()
                dq[:, j] += 0.02
                _, up_d = cup_pose(dq)
                J[:, :, c] = (up_d[:, :2] - r) / 0.02
            step = torch.linalg.solve(J + 1e-4 * torch.eye(2, device=env.device), -r[..., None])[..., 0]
            q[:, 4:6] = q[:, 4:6] + step.clamp(-0.5, 0.5)
            q = torch.minimum(torch.maximum(q, lim[:, 0]), lim[:, 1])
        pos, up = cup_pose(q)
        tilt = torch.acos(up[:, 2].clamp(-1, 1))
        ok = (tilt < 3.0 / 57.2958) & (pos >= torch.tensor(BOX_LO, device=env.device)).all(-1) & (pos <= torch.tensor(BOX_HI, device=env.device)).all(-1)
        keep_q.append(q[ok].cpu()); keep_pos.append(pos[ok].cpu()); keep_tilt.append(tilt[ok].cpu())
        print(f"batch {b}: kept {int(ok.sum())} of {env.num_envs} (tilt < 3 deg: {int((tilt < 3 / 57.2958).sum())})", flush=True)
    bank = {"q": torch.cat(keep_q), "cup_pos": torch.cat(keep_pos), "tilt": torch.cat(keep_tilt), "home": home.cpu(),
            "box_lo": BOX_LO, "box_hi": BOX_HI, "joint_names": RIGHT_ARM}
    torch.save(bank, args.out)
    cp = bank["cup_pos"]
    print(f"BANK {len(cp)} targets; glass base x {cp[:, 0].min():.2f}..{cp[:, 0].max():.2f}, y {cp[:, 1].min():.2f}..{cp[:, 1].max():.2f}, "
          f"z {cp[:, 2].min():.2f}..{cp[:, 2].max():.2f}; tilt max {bank['tilt'].max() * 57.3:.2f} deg", flush=True)
    env.close()
