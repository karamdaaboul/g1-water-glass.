"""v2 (glass in the hand, g1_inhand_task.py): copy of ../g1_water_server.py with the imports switched.
G1 water-glass env server (2026-11-18): runs G1WaterTaskEnv in the Isaac Lab venv and serves it to a learner in another
interpreter (the QC trainer's JAX venv) over a local socket.

Protocol (multiprocessing.connection, authkey from G1W_AUTHKEY): every request and reply is two byte messages, a JSON header and
an .npz payload (portable across numpy versions). Requests: {"cmd": "spec" | "obs" | "step" | "close"}; "step" carries
actions (N, 7) in the env's action units. Replies to "step": obs, rewards, dones, time_outs (truncated and not terminated),
final_observation (the pre-reset observation of finished envs; other rows = obs), costs (the spill increase of the step),
outcome (finished envs: 1 success, 2 spilled, 3 unsafe, 4 timeout; else 0; the first that fired in that order), final_spill (the
spilled fraction at the end of finished episodes), and a header "log" with scalars. The env is reset once at start-up and auto-resets finished envs inside step.

G1W_OUTCOME_LOG (optional): JSONL path; every 256 finished episodes the server appends their outcome counts and mean final spill.
G1W_VIDEO_DIR (optional, eval): record G1W_VIDEO_STEPS env steps with Isaac Lab's VideoRecorder (headless Newton GL viewer, water
particles shown), camera G1W_VIDEO_EYE / G1W_VIDEO_LOOKAT ('x,y,z') and G1W_VIDEO_FOCAL, file prefix G1W_VIDEO_PREFIX.
"""
import argparse
import io
import json
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--address", required=True, help="Unix socket path the server listens on.")
add_launcher_args(parser)
args = parser.parse_args()

from g1_inhand_env import set_mpm_capacity  # noqa: E402
from g1_inhand_task import FAIL_SPILL, SUCCESS_SPILL, G1WaterTaskEnv, G1WaterTaskEnvCfg  # noqa: E402

cfg = G1WaterTaskEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
set_mpm_capacity(cfg)
if os.environ.get("G1W_VIDEO_DIR"):
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.visualizers import VisualizerCfg, WindowCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    vec = lambda k, d: tuple(float(x) for x in os.environ.get(k, d).split(","))  # noqa: E731
    cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=vec("G1W_VIDEO_EYE", "1.35,-0.45,1.30"), lookat=vec("G1W_VIDEO_LOOKAT", "0.25,0.10,0.85"),
                                             focal_length=float(os.environ.get("G1W_VIDEO_FOCAL", "28")), window=WindowCfg(size=(1280, 720)),
                                             backend=NewtonGLBackendCfg(show_particles=True, particle_color=(0.15, 0.45, 0.95)))]
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=os.environ["G1W_VIDEO_DIR"],
                                            output_filename_prefix=os.environ.get("G1W_VIDEO_PREFIX", "policy"),
                                            video_length=int(os.environ.get("G1W_VIDEO_STEPS", "300")),
                                            fps=int(round(1.0 / (cfg.sim.dt * cfg.decimation))))]


def pack(**arrays) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def unpack(blob: bytes) -> dict:
    import numpy as np

    with np.load(io.BytesIO(blob), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


with launch_simulation(cfg, args):
    from multiprocessing.connection import Listener

    import numpy as np
    import torch

    env = G1WaterTaskEnv(cfg=cfg)
    obs_dict, _ = env.reset(seed=args.seed)
    obs = obs_dict["policy"]
    num_actions = env.action_manager.total_action_dim
    names = [n for n in env.termination_manager.active_terms]
    counts, spills, n_done, t_start = {k: 0 for k in names}, [], 0, time.time()
    outcome_log = os.environ.get("G1W_OUTCOME_LOG")

    if os.path.exists(args.address):
        os.unlink(args.address)
    listener = Listener(args.address, family="AF_UNIX", authkey=os.environ.get("G1W_AUTHKEY", "g1w").encode())
    print(f"[server] READY {args.address} envs {env.num_envs} obs {obs.shape[1]} act {num_actions} device {args.device}", flush=True)
    conn = listener.accept()
    try:
        while True:
            head = json.loads(conn.recv_bytes().decode())
            payload = unpack(conn.recv_bytes())
            cmd = head["cmd"]
            if cmd == "spec":
                conn.send_bytes(json.dumps({"num_envs": env.num_envs, "num_obs": int(obs.shape[1]), "num_actions": int(num_actions),
                                            "step_dt": env.step_dt, "max_episode_length": int(env.max_episode_length)}).encode())
                conn.send_bytes(pack())
            elif cmd == "obs":
                conn.send_bytes(b"{}")
                conn.send_bytes(pack(obs=obs.cpu().numpy()))
            elif cmd == "step":
                act = torch.as_tensor(payload["actions"], device=env.device, dtype=torch.float32)
                obs_dict, rew, term, trunc, extras = env.step(act)
                obs = obs_dict["policy"]
                done = term | trunc
                final = extras["final_observation"]
                outcome = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
                for code, k in ((4, "time_out"), (3, "unsafe"), (2, "spilled"), (1, "success")):  # the last write wins: success first
                    if k in names:
                        outcome = torch.where(done & env.termination_manager.get_term(k), code, outcome)
                log = {}
                if done.any():
                    ids = torch.nonzero(done).flatten()
                    for k in names:
                        hit = env.termination_manager.get_term(k)[ids]
                        counts[k] += int(hit.sum())
                    spills.extend(env._final_spill[ids].tolist())
                    n_done += len(ids)
                    if outcome_log and n_done >= 256:
                        row = {"time_s": round(time.time() - t_start, 1), "episodes": n_done, **{k: v / n_done for k, v in counts.items()},
                               "mean_final_spill": float(np.mean(spills)), "share_spill_le_success": float(np.mean(np.array(spills) <= SUCCESS_SPILL)),
                               "fail_spill": FAIL_SPILL}
                        with open(outcome_log, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        counts, spills, n_done = {k: 0 for k in names}, [], 0
                for k, v in extras.get("log", {}).items():
                    try:
                        log[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                conn.send_bytes(json.dumps({"log": log}).encode())
                conn.send_bytes(pack(obs=obs.cpu().numpy(), rewards=rew.cpu().numpy(), dones=done.cpu().numpy(),
                                     time_outs=(trunc & ~term).cpu().numpy(), final_observation=final.cpu().numpy(),
                                     costs=extras["costs"].cpu().numpy(), outcome=outcome.cpu().numpy(),
                                     final_spill=torch.where(done, env._final_spill, torch.zeros_like(env._final_spill)).cpu().numpy()))
            elif cmd == "close":
                conn.send_bytes(b"{}")
                conn.send_bytes(pack())
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
    finally:
        conn.close()
        listener.close()
        env.close()
        print("[server] closed", flush=True)
    sys.exit(0)
