"""Tipping-predictor data (option B, 2026-11-19; user: no execution filter, the model gives the privileged 'glass down' cost).

Each env runs one scripted hand motion past a free-standing glass per batch, with the water off (it is one-way coupled, so it
does not move the glass; its mass is in the randomized glass mass instead):
  up (to 20 cm above the table) -> over (above the start point) -> down (to the push height) -> sweep through the glass -> hold.
Randomized per env and batch: the hand point that leads the sweep (palm frame), the sweep direction (any horizontal angle),
height (2-14 cm above the table; the glass is 12 cm tall), end point (stops 6 cm before the glass axis .. 12 cm past it),
duration (0.35-3 s, i.e. ~0.05-0.6 m/s), hand open or closed, glass mass (0.25-0.55 kg: glass + water), COM height and friction.

Saved per env and policy step (15 Hz), state BEFORE the step: gpos, gquat (glass, env frame), bpos (hand bodies, env frame), q (arm),
closed; action (the 7 arm commands sent); tilt (rad) and down (tilt > 45 deg or the glass fell off the table) AFTER the step.
Positions only: tip_features.py makes the features with finite-difference velocities (Newton's velocity readback is ~3x
below the actual motion, checked on the first collection). Windows are cut in tip_train.py, split by episode.
"""
import argparse
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--batches", type=int, default=40)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args = parser.parse_args()
os.environ["G1P_NO_WATER"] = "1"

from g1_pp_env import TABLE_TOP, G1PPEnv, G1PPEnvCfg  # noqa: E402
from tip_features import HAND_BODIES  # noqa: E402

# Hand points that can lead a sweep (palm frame; fingers curl towards +y, the Dex3 ring centre is GRIP = (0.08, 0.045, 0)):
LEAD_POINTS = [(0.04, -0.03, 0.0),   # back of the hand
               (0.08, 0.045, 0.0),   # ring centre (the grasp approach)
               (0.13, 0.00, 0.0),    # finger side
               (0.02, 0.06, 0.0),    # thumb side
               (0.06, 0.0, -0.03)]   # under the palm
T_UP, T_OVER, T_DOWN, T_HOLD = 0.8, 1.2, 1.0, 1.0

cfg = G1PPEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.episode_length_s = 60.0
with launch_simulation(cfg, args):
    import numpy as np
    import torch
    from isaaclab.utils import math as mu

    env = G1PPEnv(cfg=cfg)
    n, d, dt = env.num_envs, env.device, env.step_dt
    g = torch.Generator(device=d).manual_seed(1000 + args.seed)
    term = env.action_manager.get_term("right_arm")
    ids_arm, vmax = term._joint_ids, term._max_vel[0]
    all_ids = torch.arange(n, device=d)
    zw = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1)
    origin = env.scene.env_origins
    r = env.robot
    hand_bodies = [r.body_names.index(b) for b in HAND_BODIES]
    lead_pts = torch.tensor(LEAD_POINTS, device=d)
    glass_shape_mu = None
    try:
        model = env.sim.backend_manager.backend.model
        glass_shape_mu = env.glass._root_view.get_attribute("shape_material_mu", model)[:, 0]
    except Exception as e:  # noqa: BLE001
        print(f"WARN no glass friction randomization: {e}", flush=True)

    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=g, device=d)

    def raw_state(closed):
        gp, gq = env.glass_pose_w()
        return {"gpos": (gp - origin).cpu(), "gquat": gq.cpu(), "bpos": (r.data.body_pos_w.torch[:, hand_bodies] - origin[:, None]).cpu(),
                "q": r.data.joint_pos.torch[:, ids_arm].cpu(), "closed": closed.cpu()}

    def lead_point(p_loc):
        pp, pq = env.palm_pose_w()
        return pp + mu.quat_apply(pq, p_loc), pp, pq

    KEYS = ("gpos", "gquat", "bpos", "q", "closed", "action", "tilt", "down")
    out = {k: [] for k in KEYS + ("episode", "params")}
    t0 = time.time()
    for b in range(args.batches):
        env.reset()
        # glass physics: mass 0.25-0.55 kg (glass + water), COM 3.5-7 cm up, friction 0.5-1.0
        mass = U(0.25, 0.55, n, 1)
        com = torch.zeros(n, 1, 3, device=d)
        com[:, 0, 2] = U(0.035, 0.07, n)
        env.glass.set_masses_index(masses=mass, env_ids=all_ids)
        env.glass.set_coms_index(coms=com, env_ids=all_ids)
        if glass_shape_mu is not None:
            mu_v = U(0.5, 1.0, n, 1)
            import warp as wp  # noqa: PLC0415
            from newton import ModelFlags  # noqa: PLC0415
            fv = wp.to_torch(glass_shape_mu)
            fv[:] = mu_v.expand(-1, fv.shape[1])
            env.sim.backend_manager.backend.model_changes.add(int(ModelFlags.SHAPE_PROPERTIES))
            env.sim.backend_manager.backend.notify_model_changes()
        closed = (torch.rand(n, generator=g, device=d) < 0.3).float()
        env.set_hand(all_ids[closed > 0], True)
        k_lead = torch.randint(0, len(LEAD_POINTS), (n,), generator=g, device=d)
        p_loc = lead_pts[k_lead] + U(-0.01, 0.01, n, 3)
        ang = U(0.0, 2 * np.pi, n)
        dirv = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
        h = U(0.02, 0.14, n)
        s_end = U(-0.06, 0.12, n)
        t_sweep = U(0.35, 3.0, n)
        gp0 = env.glass_pose_w()[0].clone()
        axis = gp0.clone()
        axis[:, 2] = origin[:, 2] + TABLE_TOP + h
        start_pt = axis + 0.13 * dirv
        end_pt = axis - s_end[:, None] * dirv
        clear = origin[:, 2] + TABLE_TOP + 0.20
        params = torch.stack([mass[:, 0], com[:, 0, 2], mu_v[:, 0] if glass_shape_mu is not None else torch.zeros(n, device=d),
                              closed, k_lead.float(), ang, h, s_end, t_sweep], -1)
        recs = {k: [] for k in KEYS}
        segs = []
        p_now = lead_point(p_loc)[0].clone()
        up_goal = p_now.clone(); up_goal[:, 2] = clear
        over_goal = start_pt.clone(); over_goal[:, 2] = clear
        segs = [("up", T_UP, None), ("over", T_OVER, None), ("down", T_DOWN, None), ("sweep", None, t_sweep), ("hold", T_HOLD, None)]
        for name, T, Tv in segs:
            start = lead_point(p_loc)[0].clone()
            goal = {"up": up_goal, "over": over_goal, "down": start_pt, "sweep": end_pt, "hold": start}[name]
            Te = Tv if Tv is not None else torch.full((n,), T, device=d)
            steps = int(round(float(Te.max()) / dt))
            for k in range(max(1, steps)):
                tau = ((k + 1) * dt / Te).clamp(max=1.0)[:, None]
                s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                ds = torch.where(tau < 1, (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / Te[:, None], torch.zeros_like(tau))
                p_ref, v_ref = start + (goal - start) * s, (goal - start) * ds
                b_pt, pp, pq = lead_point(p_loc)
                v = v_ref + 4.0 * (p_ref - b_pt)
                w = 4.0 * torch.cross(mu.quat_apply(pq, zw), zw, dim=-1)
                J = r.data.body_link_jacobian_w.torch[:, env.palm - 1][:, :, ids_arm]
                rr = b_pt - pp
                rx = torch.zeros(n, 3, 3, device=d)
                rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 0], rx[:, 1, 2], rx[:, 2, 0], rx[:, 2, 1] = -rr[:, 2], rr[:, 1], rr[:, 2], -rr[:, 0], -rr[:, 1], rr[:, 0]
                A = torch.cat([J[:, :3] - rx @ J[:, 3:], J[:, 3:5]], 1)
                x = torch.cat([v, w[:, :2]], -1)
                qd = (A.transpose(1, 2) @ torch.linalg.inv(A @ A.transpose(1, 2) + 0.05**2 * torch.eye(5, device=d)) @ x[..., None])[..., 0]
                act = (qd / vmax).clamp(-1.0, 1.0)
                for k_, v_ in raw_state(closed).items():
                    recs[k_].append(v_)
                recs["action"].append(act.cpu())
                env.step(act)
                gpn = env.glass_pose_w()[0]
                tilt = env.glass_tilt()
                down = (tilt > np.pi / 4) | (gpn[:, 2] - origin[:, 2] < TABLE_TOP - 0.03)
                recs["tilt"].append(tilt.cpu())
                recs["down"].append(down.cpu())
        for key in recs:
            out[key].append(torch.stack(recs[key], 1).numpy())  # (n, T, ...)
        out["episode"].append(np.arange(n) + b * n)
        out["params"].append(params.cpu().numpy())
        dn = torch.stack(recs["down"], 1).any(1).float().mean()
        print(f"BATCH {b + 1}/{args.batches}: steps {len(recs['tilt'])}, glass down in {float(dn) * 100:.0f} % of pushes, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    T_min = min(a.shape[1] for a in out["tilt"])
    save = {k: np.concatenate([a[:, :T_min] for a in v], 0) for k, v in out.items() if k in KEYS}
    save["episode"] = np.concatenate(out["episode"])
    save["params"] = np.concatenate(out["params"])
    save["dt"] = np.array(dt)
    save["table_top"] = np.array(TABLE_TOP)
    save["param_names"] = np.array(["mass", "com_z", "mu", "closed", "lead", "angle", "height", "s_end", "t_sweep"])
    np.savez_compressed(args.out, **save)
    print(f"SAVED {args.out}: {save['gpos'].shape[0]} pushes x {T_min} steps", flush=True)
    env.close()
