"""View (2026-11-18): the right Dex3 hand at the arm home pose in Isaac Lab's tri-hand gripper pose (index/middle 1.0/1.0, thumb
0/-0.4/-0.7, retargeter g1_upper_body_motion_ctrl_retargeter.py), glass moved away; one frame from above along the palm z axis.
Prints the finger link origins in the palm frame."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--out", default="hand_view"); add_launcher_args(parser); args = parser.parse_args()
from g1_hold_env import P_HOME, RIGHT_HAND, G1HoldEnv, G1HoldEnvCfg, set_mpm_capacity  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.visualizers import VisualizerCfg, WindowCfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402
cfg = G1HoldEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device; set_mpm_capacity(cfg)
x, y, z = P_HOME
cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=(x + 0.001, y, z + 0.32), lookat=(x, y, z), focal_length=24.0, window=WindowCfg(size=(1024, 1024)),
                                         backend=NewtonGLBackendCfg(show_particles=False))]
cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, output_filename_prefix="top", video_length=12, fps=15)]
GRIP_POSE = {"right_hand_index_0_joint": 1.0, "right_hand_index_1_joint": 1.0, "right_hand_middle_0_joint": 1.0, "right_hand_middle_1_joint": 1.0,
             "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": -0.4, "right_hand_thumb_2_joint": -0.7}
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu
    env = G1HoldEnv(cfg=cfg); env.reset()
    r = env.robot; h = env.hand_ids
    g = torch.tensor([[GRIP_POSE[j] for j in RIGHT_HAND]], device=env.device)
    q = r.data.joint_pos.torch.clone(); q[:, h] = g
    r.write_joint_position_to_sim_index(position=q); r.set_joint_position_target_index(target=g, joint_ids=h)
    far = torch.tensor([[2.0, 2.0, 0.5, 0, 0, 0, 1.0]], device=env.device)
    env.glass.write_root_pose_to_sim_index(root_pose=far, env_ids=torch.tensor([0], device=env.device))
    for _ in range(16):
        env.step(torch.zeros(1, 7, device=env.device))
    palm = env.palm; pp, pq = r.data.body_pos_w.torch[0, palm], r.data.body_quat_w.torch[0, palm]
    for n in [b for b in r.body_names if b.startswith("right_hand_")]:
        rel = mu.quat_apply_inverse(pq[None], (r.data.body_pos_w.torch[0, r.body_names.index(n)] - pp)[None])[0]
        print(f"LINK {n}: {[round(v, 3) for v in rel.tolist()]}", flush=True)
    print("FINGERS", [round(v, 3) for v in r.data.joint_pos.torch[0, h].tolist()], flush=True)
    env.close()
