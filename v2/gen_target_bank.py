"""Target bank v2 (2026-11-18): right-arm configurations that hold the in-hand glass upright, with the glass base position.
Robot-only env, pure kinematics (write joints + sim.forward). Per batch: sample shoulder pitch/roll/yaw, elbow and wrist yaw
around the v2 HOME; solve wrist roll and pitch by Gauss-Newton so the palm z axis (= glass axis) is vertical; glass base =
palm pose x (GRIP, -H/2). Keep tilt < 3 deg and the base inside BOX. Writes v2/target_bank.pt (q, cup_pos env frame, tilt)."""
import argparse
import os
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--num_envs", type=int, default=256); parser.add_argument("--batches", type=int, default=16)
parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_bank.pt")); add_launcher_args(parser); args = parser.parse_args()
from g1_glass_env import RIGHT_ARM, G1GlassEnvCfg  # noqa: E402
from g1_inhand_env import GRIP, H_GLASS, HOME  # noqa: E402
SPREAD = (0.6, 0.4, 0.6, 0.5, 0.0, 0.0, 0.6)
BOX_LO, BOX_HI = (0.18, -0.42, 0.62), (0.52, 0.12, 0.98)  # glass base, env frame (home base 0.30, -0.12, 0.806)
cfg = G1GlassEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import math as mu
    env = ManagerBasedRLEnv(cfg=cfg); env.reset(); r = env.scene["robot"]; d = env.device; n = env.num_envs
    ids = [r.joint_names.index(j) for j in RIGHT_ARM]; palm = r.body_names.index("right_hand_palm_link")
    lim = r.data.soft_joint_pos_limits.torch[0, ids]; lo, hi = lim[:, 0], lim[:, 1]
    off = torch.tensor([GRIP[0], GRIP[1], -0.5 * H_GLASS], device=d).expand(n, -1)
    zw = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1)
    base_q = r.data.joint_pos.torch.clone()

    def fk(qa):
        q = base_q.clone(); q[:, ids] = qa
        r.write_joint_position_to_sim_index(position=q); r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        env.sim.forward(); r.update(0.0)
        pp, pq = r.data.body_pos_w.torch[:, palm], r.data.body_quat_w.torch[:, palm]
        return pp + mu.quat_apply(pq, off) - env.scene.env_origins, mu.quat_apply(pq, zw)

    home = torch.tensor(HOME, device=d); keep_q, keep_pos, keep_tilt = [], [], []
    for b in range(args.batches):
        qa = home + (2 * torch.rand(n, 7, device=d) - 1) * torch.tensor(SPREAD, device=d)
        qa = torch.minimum(torch.maximum(qa, lo), hi)
        for _ in range(12):  # Gauss-Newton on (up_x, up_y) over wrist roll (4) and pitch (5)
            _, up = fk(qa); res = up[:, :2]; J = torch.zeros(n, 2, 2, device=d)
            for c, j in enumerate((4, 5)):
                dq = qa.clone(); dq[:, j] += 1e-3
                J[:, :, c] = (fk(dq)[1][:, :2] - res) / 1e-3
            step = torch.linalg.solve(J + 1e-4 * torch.eye(2, device=d), -res[..., None])[..., 0]
            qa[:, 4:6] = qa[:, 4:6] + step.clamp(-0.5, 0.5)
            qa = torch.minimum(torch.maximum(qa, lo), hi)
        pos, up = fk(qa)
        tilt = torch.acos(up[:, 2].clamp(-1, 1))
        ok = (tilt < 3.0 / 57.2958) & (pos >= torch.tensor(BOX_LO, device=d)).all(-1) & (pos <= torch.tensor(BOX_HI, device=d)).all(-1)
        keep_q.append(qa[ok].cpu()); keep_pos.append(pos[ok].cpu()); keep_tilt.append(tilt[ok].cpu())
        print(f"batch {b}: kept {int(ok.sum())} of {n} (tilt < 3 deg: {int((tilt < 3 / 57.2958).sum())})", flush=True)
    bank = {"q": torch.cat(keep_q), "cup_pos": torch.cat(keep_pos), "tilt": torch.cat(keep_tilt), "home": home.cpu(),
            "box_lo": BOX_LO, "box_hi": BOX_HI, "joint_names": RIGHT_ARM, "grip": GRIP}
    torch.save(bank, args.out)
    cp = bank["cup_pos"]
    print(f"BANK {len(cp)} targets; base x {cp[:, 0].min():.2f}..{cp[:, 0].max():.2f}, y {cp[:, 1].min():.2f}..{cp[:, 1].max():.2f}, "
          f"z {cp[:, 2].min():.2f}..{cp[:, 2].max():.2f}; tilt max {bank['tilt'].max() * 57.3:.2f} deg", flush=True)
    env.close()
