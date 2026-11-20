"""Film collected pick-and-place demos (2026-11-19, user: 'video of 4 demos', the G1 demos): replay each chosen episode's recorded
commands (7 arm + gripper) open loop in G1PPTaskEnv from its own layout (glass start and target read from the episode's first
observation), with the water and the target drawn. Prints, per demo, the recorded vs replayed outcome and the largest deviation
of the ring centre from the recorded path (the replay is open loop, so a large deviation would mean the video differs from
the recorded demo).

    python replay_pp_demos.py --demos demos/demos_pp_v2.npz --episodes 0 39 78 117 --video_dir <dir>
"""
import argparse
import json
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--demos", default="demos/demos_pp_v2.npz")
parser.add_argument("--episodes", type=int, nargs="+", default=[0, 39, 78, 117])
parser.add_argument("--video_dir", required=True)
parser.add_argument("--eye", type=float, nargs=3, default=(1.35, -0.75, 1.35))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.28, -0.12, 0.88))
parser.add_argument("--focal", type=float, default=22.0)
add_launcher_args(parser)
args = parser.parse_args()
os.environ["G1P_SHOW_WATER"] = "1"

import numpy as np  # noqa: E402

z = np.load(args.demos)
meta = json.loads(str(z["meta_json"]))
ix = meta["hand_obs_index"]
O, A, D = z["observations"], z["actions"], z["dones"] > 0
ends = np.flatnonzero(D)
starts = np.r_[0, ends[:-1] + 1]
eps = [(int(starts[e]), int(ends[e])) for e in args.episodes]
layouts = [(O[s, ix["glass"]:ix["glass"] + 2].copy(), O[s, ix["glass"]:ix["glass"] + 2] + O[s, ix["target"]:ix["target"] + 2]) for s, _ in eps]
total = sum(e - s + 1 for s, e in eps) + 15 * len(eps)

from g1_pp_task import G1PPTaskEnv, G1PPTaskEnvCfg  # noqa: E402

cfg = G1PPTaskEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = args.device
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.visualizers import VisualizerCfg, WindowCfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402

cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=tuple(args.eye), lookat=tuple(args.lookat), focal_length=args.focal, window=WindowCfg(size=(1280, 720)),
                                         backend=NewtonGLBackendCfg(show_particles=False))]
cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.video_dir, output_filename_prefix="demos",
                                        video_length=total, fps=15)]
with launch_simulation(cfg, args):
    import torch

    env = G1PPTaskEnv(cfg=cfg)
    d = env.device
    env.eval_bank = (torch.tensor(np.stack([l[0] for l in layouts]), device=d, dtype=torch.float32),
                     torch.tensor(np.stack([l[1] for l in layouts]), device=d, dtype=torch.float32))
    env.episode_count[:] = 0
    obs = env.reset()[0]["policy"]
    rows = []
    for k, (s, e) in enumerate(eps):
        if k > 0 and not done_last:
            obs = env.reset()[0]["policy"]
        dev, out, steps = 0.0, "unfinished", 0
        for t in range(s, e + 1):
            o2, rew, term, trunc, ex = env.step(torch.as_tensor(A[t][None], device=d, dtype=torch.float32))
            steps += 1
            done = bool((term | trunc)[0])
            cur = (ex["final_observation"] if done else o2["policy"])[0].cpu().numpy()
            ref = z["next_observations"][t]
            dev = max(dev, float(np.linalg.norm(cur[ix["ring"]:ix["ring"] + 3] - ref[ix["ring"]:ix["ring"] + 3])))
            if done:
                for key in ("success", "down", "spilled", "time_out"):
                    if bool(env.termination_manager.get_term(key)[0]):
                        out = key
                        break
                break
        done_last = done
        spill = float(env._final_spill[0]) if done else float(env.spilled_fraction()[0])
        if not done:  # hold still a moment so the video shows the end state
            for _ in range(15):
                env.step(torch.zeros(1, 8, device=d))
        rows.append(dict(demo=args.episodes[k], recorded_steps=e - s + 1, replay_steps=steps, replay_outcome=out,
                         spill_pct=round(100 * spill, 2), recorded_spill_pct=round(100 * float(z["costs"][s:e + 1, 0].sum()), 2),
                         ring_path_dev_mm=round(1000 * dev, 1),
                         start_xy=[round(float(v), 3) for v in layouts[k][0]], target_xy=[round(float(v), 3) for v in layouts[k][1]]))
        print("DEMO " + json.dumps(rows[-1]), flush=True)
    for _ in range(4):
        env.step(torch.zeros(1, 8, device=d))
    json.dump(rows, open(os.path.join(args.video_dir, "demos.json"), "w"), indent=1)
    env.close()
