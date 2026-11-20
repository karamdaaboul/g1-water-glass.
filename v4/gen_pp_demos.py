"""Pick-and-place demonstrations (2026-11-19) for the QC trainer: the scripted demonstrator of pp_demo.py (minimum-jerk grip-point
segments, palm held upright, damped least squares on the grip-point Jacobian) driving the RL task env G1PPTaskEnv, with the
gripper command as the 8th action (-1 open, +1 from the close phase until the release phase).

Each round resets all envs (random start and target spots), runs the phase schedule (carry time drawn per round from
--t_carry_lo..hi), and records every env until its first episode end. Only successful episodes are kept. Writes an .npz:
observations, actions, rewards (0/1), masks, dones, next_observations, costs (N, 4: spill, down, speed, contact label), meta_json
(with the observation layout, the arm's default joint positions and soft limits for the trainer's hard limits).
"""
import argparse
import json
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--t_carry_lo", type=float, default=2.0)
parser.add_argument("--t_carry_hi", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args = parser.parse_args()

from g1_pp_env import GLASS_Z, GRASP_H, H_GLASS  # noqa: E402
from g1_pp_task import CONTACT_D, N_ACTOR, N_CONS, N_PRIV, G1PPTaskEnv, G1PPTaskEnvCfg, obs_layout  # noqa: E402

cfg = G1PPTaskEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
with launch_simulation(cfg, args):
    import numpy as np
    import torch
    from isaaclab.utils import math as mu

    env = G1PPTaskEnv(cfg=cfg)
    n, d, dt = env.num_envs, env.device, env.step_dt
    term = env.action_manager.get_term("right_arm")
    ids_arm, vmax = term._joint_ids, term._max_vel  # arm (7) + waist (2), per-joint command scale
    nj = len(ids_arm)
    wcol = torch.tensor([1.0] * 7 + [0.3] * (nj - 7), device=d)  # weighted least norm: the waist moves only as much as needed
    zw = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1)
    origin = env.scene.env_origins
    stem_z = GLASS_Z + GRASP_H
    rng = np.random.default_rng(args.seed)
    keep = {k: [] for k in ("observations", "actions", "rewards", "masks", "dones", "next_observations", "costs", "hand_q")}
    n_ok = n_all = 0
    outcomes = {"success": 0, "down": 0, "spilled": 0, "time_out": 0, "unfinished": 0}

    def open_dir():
        _, pq = env.palm_pose_w()
        u = mu.quat_apply(pq, torch.tensor([0.7071, 0.7071, 0.0], device=d).expand(n, -1))
        u[:, 2] = 0.0
        return u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    for rnd in range(args.rounds):
        t_carry = float(rng.uniform(args.t_carry_lo, args.t_carry_hi))
        phases = [("up", 1.0), ("over", 1.5), ("down", 1.0), ("approach", 1.6), ("close", 0.5), ("lift", 1.0), ("carry", t_carry),
                  ("lower", 1.0), ("settle", 0.6), ("release", 0.5), ("rise", 0.8), ("retreat", 1.0), ("hold", 0.6)]  # up first, then back (lift-off limit)
        obs = env.reset()[0]["policy"]
        clear_z = origin[:, 2] + GLASS_Z + 0.12 + 0.06
        pre = {}
        alive = torch.ones(n, dtype=torch.bool, device=d)
        rec = [[] for _ in range(n)]
        result = ["unfinished"] * n
        grip_cmd = -1.0
        for name, T in phases:
            start = env.grip_point_w().clone()
            grasp, tgt = env.stem_mid_w(), env.target_pos_w()
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
                goal = start.clone(); goal[:, :2] = tgt[:, :2]; goal[:, 2] = origin[:, 2] + stem_z + 0.06
            elif name == "lower":
                goal = start.clone(); goal[:, 2] = origin[:, 2] + stem_z + 0.003
            elif name == "settle":
                goal = tgt.clone(); goal[:, 2] = origin[:, 2] + stem_z + 0.003
            elif name == "retreat":
                goal = start - 0.10 * open_dir()
            elif name == "rise":  # straight up 10 cm: the ring clears the glass top (12 cm) before moving back
                goal = start + torch.tensor([0.0, 0.0, 0.10], device=d)
            else:
                goal = start
            if os.environ.get("G1P_DEMO_DEBUG") and name in ("close", "release", "retreat"):
                gpt, smid = env.grip_point_w(), env.stem_mid_w()
                hz = (gpt - smid)[:, :2].norm(dim=-1) * 1000; dz = (gpt - smid)[:, 2] * 1000
                print(f"DEBUG {name}: ring horiz mm {[round(float(x)) for x in hz]}, dz mm {[round(float(x)) for x in dz]}, "
                      f"grasped {env.grasped.int().tolist()}, on_table {env.on_table().int().tolist()}, "
                      f"base z-table mm {[round(float(x)) for x in (env.glass_base_e()[:, 2] - 0.75) * 1000]}, tilt deg {[round(float(x), 1) for x in env.glass_tilt() * 57.3]}", flush=True)
            if name == "close":
                grip_cmd = 1.0
            if name == "release":
                grip_cmd = -1.0
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
                Aw = A * wcol
                qd = wcol * (Aw.transpose(1, 2) @ torch.linalg.inv(Aw @ Aw.transpose(1, 2) + 0.05**2 * torch.eye(5, device=d)) @ x[..., None])[..., 0]
                act = torch.cat([(qd / vmax).clamp(-1.0, 1.0), torch.full((n, 1), grip_cmd, device=d)], -1)
                hq_before = env.robot.data.joint_pos.torch[:, env.hand_ids].clone()  # finger angles at the observation (start bank)
                o2, rew, term_, trunc, ex = env.step(act)
                done = term_ | trunc
                nxt = torch.where(done[:, None], ex["final_observation"], o2["policy"])
                for i in torch.nonzero(alive).flatten().tolist():
                    rec[i].append((obs[i].cpu().numpy(), act[i].cpu().numpy(), float(rew[i] > 0.5), float(term_[i]), float(done[i]),
                                   nxt[i].cpu().numpy(), ex["costs"][i].cpu().numpy(), hq_before[i].cpu().numpy()))
                    if done[i]:
                        alive[i] = False
                        for key in ("success", "down", "spilled", "time_out"):
                            if bool(env.termination_manager.get_term(key)[i]):
                                result[i] = key
                                break
                obs = o2["policy"]
                if not alive.any():
                    break
            if not alive.any():
                break
        for i in range(n):
            outcomes[result[i]] += 1
            n_all += 1
            if result[i] != "success":
                continue
            n_ok += 1
            ob, ac, rw, tm, dn, nx, co, hq = map(np.array, zip(*rec[i]))
            keep["hand_q"].append(hq)
            keep["observations"].append(ob); keep["actions"].append(ac); keep["rewards"].append(rw)
            keep["masks"].append(1.0 - tm); keep["dones"].append(dn); keep["next_observations"].append(nx); keep["costs"].append(co)
        print(f"ROUND {rnd + 1}/{args.rounds}: t_carry {t_carry:.2f} s, kept {n_ok}/{n_all}, outcomes {outcomes}", flush=True)
    data = {k: np.concatenate(v).astype(np.float32) for k, v in keep.items()}
    lim = env.robot.data.soft_joint_pos_limits.torch[0, ids_arm]
    meta = {"action_bounds": [1.0] * (nj + 1), "obs_terms": ["priv", "cons", "actor"], "obs_term_dims": [N_PRIV, N_CONS, N_ACTOR],
            "actor_obs_start": N_PRIV + N_CONS, "engage_obs_start": N_PRIV, "hand_obs_index": obs_layout(),
            "q_default": env.robot.data.default_joint_pos.torch[0, ids_arm].tolist(), "q_lo": lim[:, 0].tolist(), "q_hi": lim[:, 1].tolist(),
            "arm_dims": list(range(nj)), "arm_cmd_scale": vmax.tolist(), "object_height": H_GLASS, "contact_d": CONTACT_D,
            "control_dt": float(dt), "cost_channels": ["spill", "disturb", "speed", "contact"], "push_k": __import__("g1_pp_task").PUSH_K, "water": {"xsph": __import__("g1_pp_env").PBF_XSPH, "free": __import__("g1_pp_env").FREE},
            "demonstrator": "scripted pick-and-place (gen_pp_demos.py, pp_demo.py controller)", "outcomes": outcomes}
    np.savez(args.out, meta_json=json.dumps(meta), **data)
    ep_spill = data["costs"][:, 0].sum() / max(n_ok, 1)
    print(f"SAVED {args.out}: {n_ok} episodes, {len(data['actions'])} steps, obs {data['observations'].shape[1]}, "
          f"mean spill per episode {ep_spill * 100:.2f} %, contact-label steps per episode {data['costs'][:, 3].sum() / max(n_ok, 1):.1f}", flush=True)
    env.close()
