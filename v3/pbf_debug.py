"""Debug (2026-11-19): density of the settled PBF water relative to rho0, nearest-neighbour spacing, and the settling curve."""
import torch
import warp as wp
from pbf_water import PBFWater, compute_lambda
wp.init(); dev = "cuda:0"
w = PBFWater(2, 0.04, 0.046, 0.010, 0.070, 0.050, dev, substeps=16, iterations=6)
pos = torch.zeros(2, 3, device=dev); pos[:, 0] = torch.arange(2, device=dev) * 1.0; pos[:, 2] = 1.0
quat = torch.zeros(2, 4, device=dev); quat[:, 3] = 1.0
w.reset(torch.arange(2, device=dev), pos, quat)
def stats(tag):
    w.grid.build(w.x, w.h)
    wp.launch(compute_lambda, dim=w.N, inputs=[w.grid.id, w.x, w.active, w.lam, w.rho, w.h, w.rho0, 1e2], device=dev)
    rho = wp.to_torch(w.rho).view(2, w.P)[0] / w.rho0
    x = w.positions()[0]
    d = torch.cdist(x, x); d.fill_diagonal_(1.0)
    top = x[:, 2].sort().values[-w.P // 20:].median() - 1.0 + w.rp
    print(f"{tag}: rho/rho0 median {rho.median():.3f} p90 {rho.quantile(0.9):.3f} max {rho.max():.3f}; nn spacing median {d.min(1).values.median() * 1000:.2f} mm; surface {top * 100:.2f} cm", flush=True)
stats("init")
for k in range(10):
    w.step(pos, quat, pos, quat, 1 / 15)
    if k in (0, 1, 3, 9):
        stats(f"after {(k + 1) / 15:.2f} s")
