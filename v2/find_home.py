"""Home pose for holding a glass (2026-11-18): right-arm joints such that the palm z axis is vertical (the Dex3 fingers and thumb
then wrap a vertical glass horizontally) and the grip point GRIP (palm frame, the pocket between fingers and thumb) is at P_DES.
Batched Gauss-Newton with finite differences on pure kinematics (write joints + sim.forward, no physics step), 64 random starts
around the current HOME, for palm z = +world z and -world z; prints the best solutions."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--p_des", type=float, nargs=3, default=(0.30, -0.12, 0.85))
parser.add_argument("--grip", type=float, nargs=3, default=(0.07, 0.045, 0.0))
parser.add_argument("--ref", type=float, nargs=7, default=None, help="Rank solutions by distance to this q (default HOME).")
parser.add_argument("--face_x", action="store_true", help="Robot yaw 0 (faces +x); G1_29DOF_CFG starts at yaw 90 deg (faces +y)."); add_launcher_args(parser); args = parser.parse_args()
from g1_glass_env import RIGHT_ARM, G1GlassEnvCfg  # noqa: E402
from g1_water_env import HOME  # noqa: E402
cfg = G1GlassEnvCfg(); cfg.scene.num_envs = 64; cfg.sim.device = args.device
if args.face_x:
    cfg.scene.robot = cfg.scene.robot.replace(init_state=cfg.scene.robot.init_state.replace(rot=(0.0, 0.0, 0.0, 1.0)))
with launch_simulation(cfg, args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import math as mu
    env = ManagerBasedRLEnv(cfg=cfg); env.reset(); r = env.scene["robot"]; d = env.device; n = env.num_envs
    ids = [r.joint_names.index(j) for j in RIGHT_ARM]; palm = r.body_names.index("right_hand_palm_link")
    lim = r.data.soft_joint_pos_limits.torch[0, ids]; lo, hi = lim[:, 0], lim[:, 1]
    grip = torch.tensor(args.grip, device=d).expand(n, -1); pdes = torch.tensor(args.p_des, device=d)
    base_q = r.data.joint_pos.torch.clone()

    def fk(qa):
        q = base_q.clone(); q[:, ids] = qa
        r.write_joint_position_to_sim_index(position=q); r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        env.sim.forward(); r.update(0.0)
        pp, pq = r.data.body_pos_w.torch[:, palm], r.data.body_quat_w.torch[:, palm]
        p = pp + mu.quat_apply(pq, grip) - env.scene.env_origins
        zax = mu.quat_apply(pq, torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1))
        return p, zax

    for sign in (1.0, -1.0):
        torch.manual_seed(0)
        ref = torch.tensor(args.ref if args.ref else HOME, device=d)
        qa = ref + 0.6 * (2 * torch.rand(n, 7, device=d) - 1)
        qa = torch.minimum(torch.maximum(qa, lo), hi)
        zt = torch.tensor([0.0, 0.0, sign], device=d)

        def res(qa):
            p, zax = fk(qa)
            return torch.cat([p - pdes, 0.3 * (zax - zt)], -1)  # 0.3 m per unit of axis error

        for it in range(40):
            e = res(qa); J = torch.zeros(n, 6, 7, device=d)
            for j in range(7):
                dq = qa.clone(); dq[:, j] += 1e-3; J[:, :, j] = (res(dq) - e) / 1e-3
            step = -(J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + 1e-3 * torch.eye(6, device=d), e[..., None]))[..., 0]
            qa = torch.minimum(torch.maximum(qa + step.clamp(-0.3, 0.3), lo), hi)
        p, zax = fk(qa)
        perr = (p - pdes).norm(dim=-1); tilt = torch.rad2deg(torch.acos((zax * zt).sum(-1).clamp(-1, 1)))
        dev = (qa - ref).abs().sum(-1)
        ok = (perr < 0.005) & (tilt < 1.0)
        print(f"palm z = {sign:+.0f} world z: {int(ok.sum())} of {n} solved", flush=True)
        if ok.any():
            k = torch.nonzero(ok).flatten()[torch.argmin(dev[ok])]
            print(f"   best (least change from HOME): q {[round(x, 3) for x in qa[k].tolist()]}, pos err {perr[k] * 1000:.1f} mm, tilt {tilt[k]:.2f} deg", flush=True)
    env.close()
