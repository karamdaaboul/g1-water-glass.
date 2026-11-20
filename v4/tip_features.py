"""Tipping-predictor features (2026-11-19), shared by tip_train.py and the pick-and-place task env.

Inputs are positions only, at the current and the previous policy step (velocities are finite differences over the step:
Newton's velocity readback is ~3x below the actual motion, checked on the push data, as in v2). All positions in the env frame.
  gpos (..., 3) glass base, gquat (..., 4) glass orientation (x, y, z, w), bpos (..., 9, 3) hand bodies (HAND_BODIES order),
  q (..., 7) right-arm joints, closed (...) hand closed 0/1.
Feature vector (79): glass up (3), glass lin vel (3), glass ang vel (3), glass base height above the table (1), hand bodies rel.
the glass base (27), hand body velocities (27), arm q (7), arm qd (7), hand closed (1).
"""
import torch

HAND_BODIES = ["right_wrist_yaw_link", "right_hand_palm_link", "right_hand_index_0_link", "right_hand_index_1_link",
               "right_hand_middle_0_link", "right_hand_middle_1_link", "right_hand_thumb_0_link", "right_hand_thumb_1_link",
               "right_hand_thumb_2_link"]
N_FEAT = 79


def _quat_apply(q, v):  # q (..., 4) xyzw
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _ang_vel(q0, q1, dt):
    """World angular velocity taking q0 to q1 over dt (xyzw)."""
    conj = torch.cat([-q0[..., :3], q0[..., 3:4]], -1)
    x1, y1, z1, w1 = q1.unbind(-1)
    x0, y0, z0, w0 = conj.unbind(-1)
    d = torch.stack([w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0, w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                     w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0, w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0], -1)  # q1 * conj(q0)
    d = torch.where(d[..., 3:4] < 0, -d, d)
    s = d[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    ang = 2.0 * torch.atan2(s, d[..., 3:4])
    return d[..., :3] / s * ang / dt


def tip_features(cur: dict, prev: dict, dt: float, table_top: float) -> torch.Tensor:
    z = torch.zeros_like(cur["gpos"]); z[..., 2] = 1.0
    up = _quat_apply(cur["gquat"], z)
    gv = (cur["gpos"] - prev["gpos"]) / dt
    gw = _ang_vel(prev["gquat"], cur["gquat"], dt)
    h = cur["gpos"][..., 2:3] - table_top
    rel = cur["bpos"] - cur["gpos"][..., None, :]
    bv = (cur["bpos"] - prev["bpos"]) / dt
    qd = (cur["q"] - prev["q"]) / dt
    return torch.cat([up, gv, gw, h, rel.flatten(-2), bv.flatten(-2), cur["q"], qd, cur["closed"][..., None].float()], -1)
