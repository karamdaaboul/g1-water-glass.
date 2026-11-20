"""Scripted demonstrations (2026-11-18) for the G1 water-glass task, in the QC-RLPD prior-data format.

Demonstrator (resolved rate, keeps the glass upright): the glass base follows a minimum-jerk path from its start to the target over
T ~ U(T_LO, T_HI) s, with position feedback K_P; a tilt correction w = K_R (up x z) keeps it upright (yaw free); joint rates = damped
pseudo-inverse of the 5 task rows (linear 3, angular x/y) of the glass Jacobian, plus a null-space pull K_N toward the target
configuration of target_bank.pt; command = rates / max_vel, clipped to [-1, 1].
Only successful episodes are kept (as AGXCave's demos). Writes demos.npz: observations, actions, rewards (0/1), masks, dones,
next_observations, meta_json (action_bounds, obs_terms, obs_term_dims, control_dt, demonstrator)."""
import argparse
import json

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--demos", type=int, default=160)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--t_lo", type=float, default=2.5)
parser.add_argument("--t_hi", type=float, default=4.0)
parser.add_argument("--kp", type=float, default=3.0)
parser.add_argument("--kr", type=float, default=4.0)
parser.add_argument("--kn", type=float, default=0.5)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--out", default="demos.npz")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--diag", action="store_true")
parser.add_argument("--eval_json", default=None, help="Evaluation of the demonstrator (run with G1W_EVAL_TARGETS=<seed> for "
                    "eval_qc_g1.py's fixed targets): the first --episodes_per_env episodes of each env, the same per-episode metrics; no demos")
parser.add_argument("--episodes_per_env", type=int, default=4)
parser.add_argument("--video_dir", default=None, help="Record the demonstrator with Isaac Lab's VideoRecorder (headless Newton GL "
                    "viewer, water particles shown) for --video_steps env steps; no demos are written")
parser.add_argument("--video_steps", type=int, default=300)
parser.add_argument("--replay", default=None, help="With --video_dir and --num_envs 1: replay the recorded demos of this npz open-loop "
                    "(their actions; each episode's target recovered from its first observation) instead of running the demonstrator")
parser.add_argument("--replay_first", type=int, default=0, help="First demo episode to replay.")
parser.add_argument("--focal", type=float, default=12.0, help="Video camera focal length (Isaac Lab default 12).")
parser.add_argument("--eye", type=float, nargs=3, default=(1.25, -0.55, 1.15))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.28, 0.18, 0.72))
add_launcher_args(parser)
args = parser.parse_args()

from g1_water_env import HOME, set_mpm_capacity  # noqa: E402
from g1_water_task import OBS_TERM_DIMS, OBS_TERMS, G1WaterTaskEnv, G1WaterTaskEnvCfg, cup_up  # noqa: E402

cfg = G1WaterTaskEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
set_mpm_capacity(cfg)
if args.video_dir:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.visualizers import VisualizerCfg, WindowCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=tuple(args.eye), lookat=tuple(args.lookat), window=WindowCfg(size=(1280, 720)), focal_length=args.focal,
                                             backend=NewtonGLBackendCfg(show_particles=True, particle_color=(0.15, 0.45, 0.95)))]
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.video_dir, output_filename_prefix="demonstrator",
                                            video_length=args.video_steps, fps=int(round(1.0 / (cfg.sim.dt * cfg.decimation))))]
with launch_simulation(cfg, args):
    import numpy as np
    import torch

    env = G1WaterTaskEnv(cfg=cfg)
    obs, _ = env.reset()
    obs = obs["policy"]
    N, dt, vmax = env.num_envs, env.step_dt, env.action_manager.get_term("right_arm").cfg.max_vel
    term = env.action_manager.get_term("right_arm")
    home = torch.tensor(HOME, device=env.device)
    T = args.t_lo + (args.t_hi - args.t_lo) * torch.rand(N, device=env.device)
    buf = [[] for _ in range(N)]
    kept, ends, stats = [], 0, {"success": 0, "spilled": 0, "unsafe": 0, "timeout": 0}
    ev_rows, ev_count = [], [0] * N
    o_up = sum(OBS_TERM_DIMS[: OBS_TERMS.index("cup_up")]); o_v = sum(OBS_TERM_DIMS[: OBS_TERMS.index("cup_vel")])
    if args.replay:
        with np.load(args.replay) as f:
            r_obs, r_act, r_done = f["observations"], f["actions"], f["dones"]
        o_p = sum(OBS_TERM_DIMS[: OBS_TERMS.index("cup_pos")]); o_t = sum(OBS_TERM_DIMS[: OBS_TERMS.index("target")])
        bounds_ = np.r_[0, np.flatnonzero(r_done > 0) + 1]
        r_eps = [(r_obs[a], r_act[a:b]) for a, b in zip(bounds_[:-1], bounds_[1:])][args.replay_first:]
        r_target = lambda o0: torch.as_tensor(o0[o_p:o_p + 3] + o0[o_t:o_t + 3], device=env.device) + env.home_pos  # noqa: E731
        r_i, r_k = 0, 0
        env.target_pos[0] = r_target(r_eps[0][0])
        print(f"REPLAY {args.replay}: {len(r_eps)} episodes from {args.replay_first}, lengths {[len(e[1]) for e in r_eps[:8]]}...", flush=True)
    for step in range(args.max_steps):
        t = env.episode_length_buf.float() * dt
        tau = (t / T).clamp(0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T * (tau < 1.0)
        start = torch.where((env.episode_length_buf == 0)[:, None], env.cup_pos_e(), start) if step > 0 else env.cup_pos_e()
        dp = env.target_pos - start
        p_ref, v_ref = start + dp * s[:, None], dp * ds[:, None]
        v = v_ref + args.kp * (p_ref - env.cup_pos_e())
        up = cup_up(env)
        w = args.kr * torch.cross(up, env.z_axis, dim=-1)
        J = env.robot.data.body_link_jacobian_w.torch[:, env.cup_body - 1][:, :, term._joint_ids]  # fixed base: row = body - 1
        Jt = J[:, :5]  # linear xyz + angular xy (yaw free)
        x = torch.cat([v, w[:, :2]], -1)
        A = Jt @ Jt.transpose(1, 2) + args.damping**2 * torch.eye(5, device=env.device)
        pinv = Jt.transpose(1, 2) @ torch.linalg.inv(A)
        q = env.robot.data.joint_pos.torch[:, term._joint_ids]
        null = torch.eye(7, device=env.device) - pinv @ Jt
        qd = (pinv @ x[..., None])[..., 0] + (null @ (args.kn * (env.target_q - q))[..., None])[..., 0]
        action = (qd / vmax).clamp(-1.0, 1.0)
        if args.replay:
            acts = r_eps[r_i][1]
            action = torch.as_tensor(acts[r_k] if r_k < len(acts) else np.zeros(7, np.float32), device=env.device)[None]
            r_k += 1
        if args.diag:
            if step == 0:
                D = {k: torch.zeros(N, device=env.device) for k in ('tilt', 'speed', 'angvel', 'sat')}
            D['tilt'] = torch.maximum(D['tilt'], env.cup_tilt()); D['speed'] = torch.maximum(D['speed'], env.robot.data.body_lin_vel_w.torch[:, env.cup_body].norm(dim=-1))
            D['angvel'] = torch.maximum(D['angvel'], env.robot.data.body_ang_vel_w.torch[:, env.cup_body].norm(dim=-1)); D['sat'] += (action.abs() > 0.99).any(-1).float()
        nobs, rew, terminated, truncated, extras = env.step(action)
        nobs = nobs["policy"]
        done = terminated | truncated
        next_obs = torch.where(done[:, None], extras["final_observation"], nobs)
        for i in range(N):
            buf[i].append((obs[i].cpu().numpy(), action[i].cpu().numpy(), float(rew[i]), bool(terminated[i]), bool(done[i]), next_obs[i].cpu().numpy()))
        if args.replay and bool(done[0]):
            print(f"REPLAY episode {args.replay_first + r_i}: demo length {len(r_eps[r_i][1])}, replay length {r_k}, "
                  f"{'success' if float(rew[0]) > 0.5 else 'timeout' if bool(truncated[0]) else 'failed'}, final spill {float(env._final_spill[0]):.3f}", flush=True)
            r_i, r_k = r_i + 1, 0
            env.target_pos[0] = r_target(r_eps[r_i][0])
        for i in torch.nonzero(done).flatten().tolist():
            ends += 1
            ep = buf[i]
            if args.diag:
                kind = 'success' if ep[-1][2] > 0.5 else ('timeout' if bool(truncated[i]) else 'spilled')
                print(f"EP {kind:8s} len {len(ep):3d} max tilt {D['tilt'][i] * 57.3:5.1f} deg, max speed {D['speed'][i]:.2f} m/s, max ang vel {D['angvel'][i]:.2f} rad/s, saturated steps {int(D['sat'][i])}, final spill {float(env._final_spill[i]):.3f}, target dist from home {float((env.target_pos[i] - env.home_pos).norm()) if False else -1:.2f}", flush=True)
                for k in D: D[k][i] = 0.0
            if args.eval_json and ev_count[i] < args.episodes_per_env:
                kind = "success" if ep[-1][2] > 0.5 else ("timeout" if bool(truncated[i]) else ("spilled" if env._final_spill[i] > 0.3 else "unsafe"))
                acts = np.asarray([tr[1] for tr in ep]); nos = np.asarray([tr[5] for tr in ep])
                tilt = np.degrees(np.arccos(np.clip(nos[:, o_up + 2], -1, 1))).max()
                var = np.abs(np.diff(np.vstack([np.zeros((1, 7)), acts]), axis=0)).mean(1).mean()
                v = np.vstack([ep[0][0][o_v:o_v + 3][None], nos[:, o_v:o_v + 3]]); acc = np.diff(v, axis=0) / dt; jerk = np.diff(acc, axis=0) / dt
                ev_rows.append(dict(env=i, episode=ev_count[i], outcome=kind, success=kind == "success", steps=len(ep), time_s=round(len(ep) * dt, 3),
                                    final_spill=float(env._final_spill[i]), max_tilt_deg=round(float(tilt), 2), variation=float(var),
                                    acc_mean=float(np.linalg.norm(acc, axis=1).mean()), acc_max=float(np.linalg.norm(acc, axis=1).max()),
                                    jerk_mean=float(np.linalg.norm(jerk, axis=1).mean()) if len(jerk) else 0.0))
                ev_count[i] += 1
            if ep[-1][2] > 0.5:
                stats["success"] += 1
                kept.append(ep)
            elif bool(truncated[i]):
                stats["timeout"] += 1
            elif env._final_spill[i] > 0.3:
                stats["spilled"] += 1
            else:
                stats["unsafe"] += 1
            buf[i] = []
            T[i] = args.t_lo + (args.t_hi - args.t_lo) * torch.rand((), device=env.device)
        obs = nobs
        if step % 100 == 0:
            print(f"step {step}: episodes {ends}, {stats}", flush=True)
        if args.video_dir:
            if step >= args.video_steps + 5:  # a few extra steps so the clip is flushed
                print(f"VIDEO {args.video_steps} steps, episodes {ends} ({stats}) -> {args.video_dir}", flush=True)
                env.close()
                raise SystemExit(0)
            continue
        if (not args.eval_json and len(kept) >= args.demos) or (args.eval_json and min(ev_count) >= args.episodes_per_env):
            break
    if args.eval_json:
        ok = [r for r in ev_rows if r["success"]]
        m = lambda k, rs: float(np.mean([r[k] for r in rs])) if rs else None  # noqa: E731
        summary = {"policy": "demonstrator", "episodes": len(ev_rows), "success": m("success", ev_rows),
                   "outcomes": {k: sum(r["outcome"] == k for r in ev_rows) for k in ("success", "spilled", "unsafe", "timeout")},
                   "final_spill_mean": m("final_spill", ev_rows), "final_spill_success_mean": m("final_spill", ok),
                   "time_s_success_mean": m("time_s", ok), "max_tilt_deg_mean": m("max_tilt_deg", ev_rows), "variation_mean": m("variation", ev_rows),
                   "acc_mean": m("acc_mean", ev_rows), "acc_max_mean": m("acc_max", ev_rows), "jerk_mean": m("jerk_mean", ev_rows), "num_envs": N}
        print("[eval] " + json.dumps(summary), flush=True)
        with open(args.eval_json, "w") as f:
            json.dump({"summary": summary, "episodes": sorted(ev_rows, key=lambda r: (r["episode"], r["env"]))}, f, indent=1)
        env.close()
        raise SystemExit(0)
    kept = kept[: args.demos]
    cat = lambda k: np.asarray([tr[k] for ep in kept for tr in ep])
    data = dict(observations=cat(0).astype(np.float32), actions=cat(1).astype(np.float32), rewards=cat(2).astype(np.float32),
                masks=(1.0 - cat(3)).astype(np.float32), dones=cat(4).astype(np.float32), next_observations=cat(5).astype(np.float32))
    meta = dict(action_bounds=[1.0] * 7, obs_terms=OBS_TERMS, obs_term_dims=OBS_TERM_DIMS, control_dt=float(dt),
                demonstrator=f"resolved-rate upright carry, min-jerk glass path T ~ U({args.t_lo}, {args.t_hi}) s, kp {args.kp}, kr {args.kr}, kn {args.kn}",
                episodes=len(kept), attempts=ends, outcomes=stats)
    np.savez(args.out, meta_json=json.dumps(meta), **data)
    lens = [len(ep) for ep in kept]
    print(f"DEMOS {len(kept)} kept of {ends} episodes ({stats}); transitions {len(data['actions'])}, length mean {np.mean(lens):.0f} "
          f"steps; obs dim {data['observations'].shape[1]}", flush=True)
    env.close()
