"""Probe (2026-11-19): which way does the G1 face? Lateral axis = left shoulder - right shoulder; forward = lateral x up."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_pp_env import G1PPEnv, G1PPEnvCfg  # noqa: E402
cfg = G1PPEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device
with launch_simulation(cfg, args):
    import torch
    env = G1PPEnv(cfg=cfg); env.reset()
    r = env.robot; bp = r.data.body_pos_w.torch[0] - env.scene.env_origins[0]
    get = lambda n: bp[r.body_names.index(n)]
    ls, rs = get("left_shoulder_pitch_link"), get("right_shoulder_pitch_link")
    lat = ls - rs; lat[2] = 0; lat = lat / lat.norm()
    fwd = torch.linalg.cross(lat, torch.tensor([0.0, 0.0, 1.0], device=env.device))
    print(f"FACING forward {[round(v, 3) for v in fwd.tolist()]} (yaw {float(torch.atan2(fwd[1], fwd[0])) * 57.3:.0f} deg); left shoulder {[round(v, 3) for v in ls.tolist()]}, right {[round(v, 3) for v in rs.tolist()]}", flush=True)
    print("ROOT quat", [round(v, 3) for v in r.data.root_quat_w.torch[0].tolist()], "init rot", cfg.scene.robot.init_state.rot, flush=True)
    env.close()
