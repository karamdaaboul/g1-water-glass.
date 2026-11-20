"""Probe (2026-11-19): where did the GraspWeld joint go in the Newton model? joints/bodies/equality rows mentioning Glass/Grasp."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
import g1_pp_env as P  # noqa: E402
P.G1PPEnv._find_welds = lambda self: None
P.G1PPEnv.set_weld = lambda self, ids, on: None
cfg = P.G1PPEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device
with launch_simulation(cfg, args):
    env = P.G1PPEnv(cfg=cfg)
    m = env.sim.backend_manager.backend.model
    jl = list(m.joint_label); bl = list(m.body_label)
    print("JOINTS w/ Grasp/Glass:", [(i, l, int(m.joint_type.numpy()[i])) for i, l in enumerate(jl) if "Grasp" in l or "Glass" in l], flush=True)
    print("BODIES w/ Glass:", [(i, l) for i, l in enumerate(bl) if "Glass" in l], flush=True)
    print("ARTICULATIONS:", int(m.articulation_count), "eq count", int(getattr(m, "equality_constraint_count", -1)) if hasattr(m, "equality_constraint_count") else "n/a",
          "mujoco eq count", int(m.mujoco.equality_constraint_count) if hasattr(m, "mujoco") else "n/a", flush=True)
    import newton
    print("FLAG ENUMS:", [n for n in dir(newton) if "Flag" in n or "Notify" in n], flush=True)
    env.close()
