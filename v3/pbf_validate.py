"""Validate pbf_water.py (2026-11-19) against hydrostatics, standalone (no Isaac Lab app): 8 cm bowl, 6 cm cavity, water 5 cm
(1 cm free). Static theory: spill at tilt atan(free/R) = 14.0 deg, or steady lateral acceleration g free / R = 2.45 m/s^2.
Envs 0-6: settle 0.5 s, tilt (ramp 1 s about x, hold 1.5 s). Envs 7-10: settle, accelerate along +x at A for 0.5 s, then decelerate
to rest at the same rate. Then the step time for 64 envs."""
import os
import time
import torch
import warp as wp
from pbf_water import PBFWater

wp.init()
dev = "cuda:0"
R_IN, R_OUT, BOTTOM, H = (float(os.environ.get(k, v)) for k, v in (('R_IN', '0.04'), ('R_OUT', '0.046'), ('BOTTOM', '0.010'), ('H', '0.070')))
FILL = H - BOTTOM - 0.01
FREE = float(os.environ.get('FREE', '0.01'))  # target free space after settling
TILTS = [int(x) for x in os.environ.get('TILTS', '6,10,12,14,16,20,25').split(',')]
ACCS = [1.0, 2.0, 3.0, 4.0]
E = len(TILTS) + len(ACCS)
DT = 1.0 / 15.0
KW = dict(substeps=int(os.environ.get('SUB', '16')), iterations=int(os.environ.get('IT', '3')),
          xsph_c=float(os.environ.get('XSPH', '0.01')))  # XSPH viscosity (2026-11-19: rest-jitter sweep)
w = PBFWater(E, R_IN, R_OUT, BOTTOM, H, FILL, dev, **KW)
pos = torch.zeros(E, 3, device=dev); pos[:, 0] = torch.arange(E, device=dev) * 1.0; pos[:, 2] = 1.0
quat = torch.zeros(E, 4, device=dev); quat[:, 3] = 1.0
w.presettle(H - BOTTOM - FREE)
w.reset(torch.arange(E, device=dev), pos, quat)
print(f"particles per glass {w.P}, rest density {w.rho0:.3e}", flush=True)
for _ in range(8):  # settle 0.53 s
    w.step(pos, quat, pos, quat, DT)
x = w.positions()
top = torch.stack([x[e, :, 2].sort().values[-max(1, w.P // 20):].median() for e in range(E)]) - pos[:, 2] + w.rp
print(f"settled: water surface {top.mean() * 100:.2f} cm above the glass base (target {(H - FREE) * 100:.2f}, rim {H * 100:.2f}); "
      f"spilled {w.spilled_fraction().max():.4f}", flush=True)
vel = torch.zeros(E, 3, device=dev)
ang = torch.zeros(E, device=dev)
T = int(3.0 / DT)
for k in range(T):
    t = (k + 1) * DT
    p1, q1 = pos.clone(), quat.clone()
    for i, a in enumerate(TILTS):  # tilt about x through the glass base
        th = min(1.0, t / 1.0) * a / 57.2958
        q1[i] = torch.tensor([torch.sin(torch.tensor(th / 2)), 0.0, 0.0, torch.cos(torch.tensor(th / 2))], device=dev)
    for j, A in enumerate(ACCS):
        i = len(TILTS) + j
        acc = A if t <= 0.5 else (-A if t <= 1.0 else 0.0)
        vel[i, 0] = vel[i, 0] + acc * DT
        if t > 1.0:
            vel[i, 0] = 0.0
        p1[i] = pos[i] + vel[i] * DT
    w.step(pos, quat, p1, q1, DT)
    pos, quat = p1, q1
sp = w.spilled_fraction()
for i, a in enumerate(TILTS):
    print(f"TILT {a:4.1f} deg (theory spills above 14.0): spilled {sp[i] * 100:6.2f} %", flush=True)
for j, A in enumerate(ACCS):
    print(f"ACC  {A:.1f} m/s^2 for 0.5 s then brake (theory spills above 2.45 steady): spilled {sp[len(TILTS) + j] * 100:6.2f} %", flush=True)
# speed
E2 = 64
w2 = PBFWater(E2, R_IN, R_OUT, BOTTOM, H, FILL, dev, **KW)
p2 = torch.zeros(E2, 3, device=dev); p2[:, 0] = torch.arange(E2, device=dev) * 1.0; p2[:, 2] = 1.0
q2 = torch.zeros(E2, 4, device=dev); q2[:, 3] = 1.0
w2.presettle(H - BOTTOM - FREE)
w2.reset(torch.arange(E2, device=dev), p2, q2)
for _ in range(3):
    w2.step(p2, q2, p2, q2, DT)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10):
    w2.step(p2, q2, p2, q2, DT)
torch.cuda.synchronize(); el = (time.time() - t0) / 10
print(f"SPEED {E2} glasses x {w2.P} particles: {el * 1000:.1f} ms per 1/15 s policy step -> {E2 / el:.0f} env-steps/s for the water alone", flush=True)
