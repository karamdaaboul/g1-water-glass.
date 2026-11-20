"""Partial-progress start bank for the G1 pick-and-place (option 3, prepared 2026-11-19; used only if the push cost (option 5)
does not stop the policy from pushing the glass). As the excavator's reset bank: half of the training resets start from
states of the demonstrations, so the critic sees carrying and placing succeed from the first updates.

Per demo episode four starts (arm joints, hand open / closed, glass held or free, glass pose, target):
  pregrasp   8 steps before the first held step: hand open around the free glass, at grasp height
  grasped   15 steps after it: glass held, lifted
  carry      halfway between grasping and releasing: glass held on the way to the target
  lower     10 steps before the release: glass held over the target, being set down
The arm starts at rest (velocities 0), the water settled in the glass. Writes an npz the task env reads (G1P_START_BANK).

    python make_start_bank.py --demos demos/demos_pp_v3.npz --out demos/start_bank_v3.npz
"""
import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--demos", default="demos/demos_pp_v3.npz")
parser.add_argument("--out", default="demos/start_bank_v3.npz")
args = parser.parse_args()

z = np.load(args.demos)
meta = json.loads(str(z["meta_json"]))
ix = meta["hand_obs_index"]
O, D = z["observations"], z["dones"] > 0
qdef = np.asarray(meta["q_default"], np.float32)
ends = np.flatnonzero(D)
starts = np.r_[0, ends[:-1] + 1]
TYPES = ["pregrasp", "grasped", "carry", "lower"]
rows = {k: [] for k in ("q", "closed", "held", "glass", "up", "target", "type", "demo")}
for e, (s, t_end) in enumerate(zip(starts, ends)):
    held = O[s:t_end + 1, ix["held"]] > 0.5
    h = np.flatnonzero(held)
    if len(h) < 30:
        continue
    g0, g1 = s + h[0], s + h[-1]
    picks = {"pregrasp": g0 - 8, "grasped": g0 + 15, "carry": (g0 + g1) // 2, "lower": g1 - 10}
    for kind, t in picks.items():
        o = O[t]
        rows["q"].append(o[ix["q"]:ix["q"] + len(qdef)] + qdef)
        rows["closed"].append(o[ix["closed"]] > 0.5)
        rows["held"].append(o[ix["held"]] > 0.5)
        rows["glass"].append(o[ix["glass"]:ix["glass"] + 3])
        rows["up"].append(o[ix["up"]:ix["up"] + 3])
        rows["target"].append(o[ix["glass"]:ix["glass"] + 2] + o[ix["target"]:ix["target"] + 2])
        rows["type"].append(TYPES.index(kind))
        rows["demo"].append(e)
bank = {k: np.asarray(v) for k, v in rows.items()}
for k in ("q", "glass", "up", "target"):
    bank[k] = bank[k].astype(np.float32)
np.savez(args.out, types=np.array(TYPES), **bank)
held_ok = [(bank["held"][bank["type"] == i].mean(), bank["closed"][bank["type"] == i].mean()) for i in range(4)]
print(f"bank {args.out}: {len(bank['type'])} starts from {len(set(bank['demo']))} demos; per type (held, closed): "
      + ", ".join(f"{t} ({a:.2f}, {b:.2f})" for t, (a, b) in zip(TYPES, held_ok)))
