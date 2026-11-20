"""Tilt check (2026-11-19): wine glass at home; env i tilts the glass to ANGLES[i] about a horizontal axis over 1 s (resolved-rate,
glass position held), holds 1.5 s; spill and the reached tilt per env."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_wine_pbf_env import G1InHandEnv, G1InHandEnvCfg, set_mpm_capacity  # noqa: E402
ANGLES = tuple(float(x) for x in __import__('os').environ.get('TILTS', '8,12,16,20').split(','))
cfg = G1InHandEnvCfg(); cfg.scene.num_envs = len(ANGLES); cfg.sim.device = args.device; set_mpm_capacity(cfg); cfg.episode_length_s = 60.0
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu
    env = G1InHandEnv(cfg=cfg); env.reset()
    d, dt, n = env.device, env.step_dt, env.num_envs
    term = env.action_manager.get_term("right_arm"); ids = term._joint_ids; vmax = term._max_vel[0]
    for _ in range(5):
        env.step(torch.zeros(n, 7, device=d))
    p0 = env.cup_pose_w()[0].clone(); q0 = env.cup_pose_w()[1].clone()
    target = torch.tensor(ANGLES, device=d) / 57.2958
    peak = torch.zeros(n, device=d)
    for k in range(int(2.5 / dt)):
        a = torch.clamp(torch.tensor(k * dt / 1.0, device=d), max=1.0) * target  # tilt ramps over 1 s
        qd_rot = mu.quat_from_angle_axis(a, torch.tensor([1.0, 0.0, 0.0], device=d).expand(n, -1))
        q_ref = mu.quat_mul(qd_rot, q0)
        p, q = env.cup_pose_w()
        err = mu.axis_angle_from_quat(mu.quat_mul(q_ref, mu.quat_conjugate(q)))
        x = torch.cat([3.0 * (p0 - p), 4.0 * err], -1)
        J = env.robot.data.body_link_jacobian_w.torch[:, env.cup_body - 1][:, :, ids]
        qd = (J.transpose(1, 2) @ torch.linalg.inv(J @ J.transpose(1, 2) + 0.05**2 * torch.eye(6, device=d)) @ x[..., None])[..., 0]
        env.step((qd / vmax).clamp(-1, 1))
        peak = torch.maximum(peak, env.cup_tilt())
    sp = env.spilled_fraction()
    for i, ang in enumerate(ANGLES):
        print(f"TILT target {ang:4.1f} deg: reached {float(peak[i]) * 57.3:5.1f} deg, spilled {float(sp[i]) * 100:5.2f} %", flush=True)
    env.close()
