"""Water rest-jitter sweep (2026-11-19, user: 'try 3'): for XSPH viscosity c x solver iterations, the particle speed of water
at rest in an exactly fixed normal glass (6 x 12 cm, as v4) and the solver time per policy step for 64 glasses."""
import os
import sys
import time

import torch
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pbf_water import PBFWater  # noqa: E402

R_IN, WALL, CAVITY, BOTTOM, FREE = 0.026, 0.004, 0.114, 0.006, 0.01
R_OUT, H = R_IN + WALL, BOTTOM + CAVITY
dev, dt = "cuda:0", 1.0 / 15.0
SETTINGS = [(0.01, 6), (0.05, 6), (0.1, 6), (0.2, 6), (0.01, 12), (0.1, 10), (0.2, 10)]
for c, it in SETTINGS:
    w = PBFWater(4, R_IN, R_OUT, BOTTOM, H, CAVITY - FREE, dev, spacing=0.005, substeps=16, iterations=it, xsph_c=c)
    w.presettle(CAVITY - FREE)
    pos = torch.zeros(4, 3, device=dev); pos[:, 0] = torch.arange(4, device=dev) * 2.0
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev).repeat(4, 1)
    w.reset(torch.arange(4, device=dev), pos, quat)
    rms = []
    for k in range(45):  # 3 s
        w.step(pos, quat, pos, quat, dt)
        v = wp.to_torch(w.v).view(w.E, w.P, 3).norm(dim=-1)
        rms.append(float(v.pow(2).mean().sqrt()))
    level = float(wp.to_torch(w.x).view(w.E, w.P, 3)[..., 2].max(-1).values.mean())  # top particle height (glass frame)
    w64 = PBFWater(64, R_IN, R_OUT, BOTTOM, H, CAVITY - FREE, dev, spacing=0.005, substeps=16, iterations=it, xsph_c=c)
    w64.presettle(CAVITY - FREE)
    p64 = torch.zeros(64, 3, device=dev); p64[:, 0] = torch.arange(64, device=dev) * 2.0
    q64 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev).repeat(64, 1)
    w64.reset(torch.arange(64, device=dev), p64, q64)
    for _ in range(3):
        w64.step(p64, q64, p64, q64, dt)
    wp.synchronize(); t0 = time.time()
    for _ in range(20):
        w64.step(p64, q64, p64, q64, dt)
    wp.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    print(f"SWEEP xsph {c:.2f} iters {it:2d}: rest speed rms 0.5 s {rms[7]:.4f}  1 s {rms[14]:.4f}  3 s {sum(rms[-15:]) / 15:.4f} m/s; "
          f"top particle {level * 100:.2f} cm; solver {ms:.1f} ms per step (64 glasses)", flush=True)
