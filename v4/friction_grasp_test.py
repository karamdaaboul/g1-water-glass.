"""Can the Dex3 hold the full glass by friction alone? (2026-11-19, user item 4: replace the weld by a real grasp.)

The scripted controller of gen_pp_demos.py (approach, close, lift 6 cm, carry 2.5 s, hold) runs with the weld disabled
(G1P_NO_WELD=1). Per env it reports the glass's slip in the palm frame during lift + carry (cm), whether it was lifted with the
hand (glass base > 3 cm above the table at the end of the lift), dropped (glass back on the table or tilted > 45 deg while the
hand is up), NaNs, and the spill. Hand gains via G1P_HAND_KP, closing via G1P_HAND_SCALE."""
import argparse
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
add_launcher_args(parser)
args = parser.parse_args()
os.environ["G1P_NO_WELD"] = "1"

from g1_pp_env import GLASS_Z, GRASP_H, TABLE_TOP  # noqa: E402
from g1_pp_task import G1PPTaskEnv, G1PPTaskEnvCfg  # noqa: E402

cfg = G1PPTaskEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu

    env = G1PPTaskEnv(cfg=cfg)
    n, d, dt = env.num_envs, env.device, env.step_dt
    term = env.action_manager.get_term("right_arm")
    ids_arm, vmax = term._joint_ids, term._max_vel  # per joint (arm 1, waist 0.5 rad/s)
    zw = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1)
    origin = env.scene.env_origins
    env.reset()

    def open_dir():
        _, pq = env.palm_pose_w()
        u = mu.quat_apply(pq, torch.tensor([0.7071, 0.7071, 0.0], device=d).expand(n, -1)); u[:, 2] = 0.0
        return u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def glass_in_palm():
        pp, pq = env.palm_pose_w(); gp, _ = env.glass_pose_w()
        return mu.quat_apply_inverse(pq, gp - pp)

    clear_z = origin[:, 2] + GLASS_Z + 0.18
    pre = {}
    rec = {}
    phases = [("up", 1.0), ("over", 1.5), ("down", 1.0), ("approach", 1.6), ("close", 0.6), ("lift", 1.0), ("carry", 2.5), ("hold", 1.0)]
    nan = torch.zeros(n, dtype=torch.bool, device=d)
    for name, T in phases:
        start = env.grip_point_w().clone()
        grasp, tgt = env.stem_mid_w(), env.target_pos_w()
        goal = {"up": None, "over": None}.get(name, start)
        if name == "up":
            goal = start.clone(); goal[:, 2] = clear_z
        elif name == "over":
            pre["p"] = grasp - 0.12 * open_dir(); goal = pre["p"].clone(); goal[:, 2] = clear_z
        elif name == "down":
            goal = pre["p"]
        elif name == "approach":
            goal = grasp
        elif name == "lift":
            goal = start + torch.tensor([0.0, 0.0, 0.06], device=d)
        elif name == "carry":
            goal = start.clone(); goal[:, :2] = tgt[:, :2]
        grip = 1.0 if name in ("close", "lift", "carry", "hold") else -1.0
        if name == "close":
            rec["xy_before_close"] = env.glass_base_e()[:, :2].clone()
        if name == "lift":
            rec["g0"] = glass_in_palm().clone()
            rec["shove"] = (env.glass_base_e()[:, :2] - rec["xy_before_close"]).norm(dim=-1)
        for k in range(max(1, int(round(T / dt)))):
            tau = min(1.0, (k + 1) * dt / T)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T if tau < 1 else 0.0
            p_ref, v_ref = start + (goal - start) * s, (goal - start) * ds
            g = env.grip_point_w()
            v = v_ref + 5.0 * (p_ref - g)
            pp, pq = env.palm_pose_w()
            w = 4.0 * torch.cross(mu.quat_apply(pq, zw), zw, dim=-1)
            J = env.robot.data.body_link_jacobian_w.torch[:, env.palm - 1][:, :, ids_arm]
            r = g - pp
            rx = torch.zeros(n, 3, 3, device=d)
            rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 0], rx[:, 1, 2], rx[:, 2, 0], rx[:, 2, 1] = -r[:, 2], r[:, 1], r[:, 2], -r[:, 0], -r[:, 1], r[:, 0]
            A = torch.cat([J[:, :3] - rx @ J[:, 3:], J[:, 3:5]], 1)
            x = torch.cat([v, w[:, :2]], -1)
            qd = (A.transpose(1, 2) @ torch.linalg.inv(A @ A.transpose(1, 2) + 0.05**2 * torch.eye(5, device=d)) @ x[..., None])[..., 0]
            act = torch.cat([(qd / vmax).clamp(-1.0, 1.0), torch.full((n, 1), grip, device=d)], -1)
            env.step(act)
            nan |= ~torch.isfinite(env.glass_pose_w()[0]).all(-1)
        if name == "lift":
            rec["lift_z"] = (env.glass_base_e()[:, 2] - TABLE_TOP).clone()
            rec["g_lift_end"] = glass_in_palm().clone()
    slip = (glass_in_palm() - rec["g0"]).norm(dim=-1)
    z_end = env.glass_base_e()[:, 2] - TABLE_TOP
    tilt = env.glass_tilt() * 180 / math.pi
    for i in range(n):
        print(f"ENV {i}: lifted {float(rec['lift_z'][i]) * 100:.1f} cm at the end of the lift, at the end {float(z_end[i]) * 100:.1f} cm above the table, "
              f"slip in the palm {float(slip[i]) * 100:.1f} cm, tilt {float(tilt[i]):.0f} deg, spill {float(env.spilled_fraction()[i]) * 100:.1f} %, nan {bool(nan[i])}", flush=True)
    held = (z_end > 0.03) & (slip < 0.02) & (tilt < 20) & ~nan
    eq = rec["g0"][~nan].mean(0)
    lift_slip = (rec["g_lift_end"] - rec["g0"]).norm(dim=-1)  # while lifting 6 cm
    carry_slip = (glass_in_palm() - rec["g_lift_end"]).norm(dim=-1)  # carry + hold, after the lift
    ok = ~nan
    print(f"SLIP: during the 6 cm lift mean {float(lift_slip[ok].mean()) * 100:.2f} cm (max {float(lift_slip[ok].max()) * 100:.2f}), during carry + hold "
          f"mean {float(carry_slip[ok].mean()) * 100:.2f} cm (max {float(carry_slip[ok].max()) * 100:.2f}); lifted at the end of the lift mean "
          f"{float(rec['lift_z'][ok].mean()) * 100:.1f} cm of 6; dropped {int(((z_end < 0.01) & ok).sum())}/{int(ok.sum())}", flush=True)
    print(f"CLOSED-HAND CENTRE: glass axis in the palm frame after closing {eq[0]:.4f}, {eq[1]:.4f}, {eq[2]:.4f} (ring GRIP {os.environ.get('G1P_GRIP', '0.08,0.045')}); "
          f"shove while closing mean {float(rec['shove'].mean()) * 100:.2f} cm, max {float(rec['shove'].max()) * 100:.2f} cm", flush=True)
    print(f"FRICTION GRASP hand_kp {os.environ.get('G1P_HAND_KP', '20')} scale {os.environ.get('G1P_HAND_SCALE', '0.85')} "
          f"cone {os.environ.get('G1P_CONE', 'pyramidal')} impratio {os.environ.get('G1P_IMPRATIO', '1')} effort {os.environ.get('G1P_HAND_EFFORT', 'none')} "
          f"mass {os.environ.get('G1P_GLASS_MASS', '0.20')}: "
          f"held through lift + carry {int(held.sum())}/{n}, mean slip {float(slip[~nan].mean()) * 100:.1f} cm", flush=True)
    env.close()
