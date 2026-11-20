"""Probe (2026-11-18): Newton shape labels, types and bodies of the right Dex3 hand in the current water env (env 0)."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args()
from g1_water_env import set_mpm_capacity  # noqa: E402
from g1_water_task import G1WaterTaskEnv, G1WaterTaskEnvCfg  # noqa: E402
cfg = G1WaterTaskEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device; set_mpm_capacity(cfg)
with launch_simulation(cfg, args):
    import newton
    env = G1WaterTaskEnv(cfg=cfg); env.reset()
    m = env.sim.backend_manager.backend.model
    labels = list(m.shape_label); types = m.shape_type.numpy(); body = m.shape_body.numpy(); blabels = list(m.body_label)
    names = {int(v): k for k, v in vars(newton.GeoType).items() if isinstance(v, int) or hasattr(v, "value")} if hasattr(newton, "GeoType") else {}
    print("SHAPES", len(labels), flush=True)
    for i, l in enumerate(labels):
        if "right_hand" in l or "Cup" in l or "right_wrist" in l:
            print(f"   {i}: {l} | type {int(types[i])} | body {blabels[body[i]] if body[i] >= 0 else -1}", flush=True)
    env.close()
