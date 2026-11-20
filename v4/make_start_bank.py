"""Partial-progress start bank for the G1 pick-and-place (option 3, prepared 2026-11-19; used only if the push cost (option 5)
does not stop the policy from pushing the glass). As the excavator's reset bank: half of the training resets start from
states of the demonstrations, so the critic sees carrying and placing succeed from the first updates.

Per demo episode four starts (arm joints, hand open / closed, glass held or free, glass pose, target):
  pregrasp   8 steps before the first held step: hand open around the free glass, at grasp height
  grasped   15 steps after it: glass held, lifted
  carry      halfway between grasping and releasing: glass held on the way to the target
  lower     10 steps before the release: glass held over the target, being set down
The arm starts at rest (velocities 0), the water settled in the glass. Writes an npz the task env reads (G1P_START_BANK).

--recovery (2026-11-20, user: starts for the failures of the friction-grasp policy) adds per demo:
  slipped       the hand closed as while carrying (grasped pick on even demos, carry pick on odd ones), the glass standing
                upright on the table below it, NOT held: reopen, lower and grasp again (the policy had kept a closed hand
                above a glass that slid out)
  near_miss     the lower state with the target moved 2-5 cm in a random direction: correct before releasing
  far_pregrasp  a second copy of the pregrasp start when the glass stands at x >= 0.37 (far-edge grasps closed too high)

    python make_start_bank.py --demos demos/demos_pp_v3.npz --out demos/start_bank_v3.npz
"""
import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--demos", default="demos/demos_pp_v3.npz")
parser.add_argument("--out", default="demos/start_bank_v3.npz")
parser.add_argument("--recovery", action="store_true", help="add slipped, near_miss and far_pregrasp starts")
parser.add_argument("--seed", type=int, default=0, help="for the near-miss target offsets")
args = parser.parse_args()

z = np.load(args.demos)
meta = json.loads(str(z["meta_json"]))
ix = meta["hand_obs_index"]
O, D = z["observations"], z["dones"] > 0
qdef = np.asarray(meta["q_default"], np.float32)
ends = np.flatnonzero(D)
starts = np.r_[0, ends[:-1] + 1]
TYPES = ["pregrasp", "grasped", "carry", "lower"] + (["slipped", "near_miss", "far_pregrasp"] if args.recovery else [])
TABLE_Z = 0.75  # g1_pp_env.TABLE_TOP: a glass standing on the table has its base here
rng = np.random.default_rng(args.seed)
HQ = z["hand_q"] if "hand_q" in z.files else None  # finger angles (friction grasp, demos v5+)
rows = {k: [] for k in ("q", "closed", "held", "glass", "up", "target", "type", "demo") + (("hand_q",) if HQ is not None else ())}
for e, (s, t_end) in enumerate(zip(starts, ends)):
    held = O[s:t_end + 1, ix["held"]] > 0.5
    h = np.flatnonzero(held)
    if len(h) < 30:
        continue
    g0, g1 = s + h[0], s + h[-1]
    picks = {"pregrasp": g0 - 8, "grasped": g0 + 15, "carry": (g0 + g1) // 2, "lower": g1 - 10}
    extra = {}
    if args.recovery:
        extra = {"slipped": picks["grasped"] if e % 2 == 0 else picks["carry"], "near_miss": picks["lower"]}
        if O[picks["pregrasp"], ix["glass"]] >= 0.37:
            extra["far_pregrasp"] = picks["pregrasp"]
    for kind, t in list(picks.items()) + list(extra.items()):
        o = O[t]
        rows["q"].append(o[ix["q"]:ix["q"] + len(qdef)] + qdef)
        glass, up = o[ix["glass"]:ix["glass"] + 3].copy(), o[ix["up"]:ix["up"] + 3].copy()
        target = o[ix["glass"]:ix["glass"] + 2] + o[ix["target"]:ix["target"] + 2]
        held = o[ix["held"]] > 0.5
        if kind == "slipped":  # the glass slid out: upright on the table under the closed hand
            glass[2], up, held = TABLE_Z, np.array([0.0, 0.0, 1.0], np.float32), False
        if kind == "near_miss":
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.02, 0.05)
            target = target + r * np.array([np.cos(a), np.sin(a)], np.float32)
        rows["closed"].append(o[ix["closed"]] > 0.5)
        rows["held"].append(held)
        rows["glass"].append(glass)
        rows["up"].append(up)
        rows["target"].append(target)
        rows["type"].append(TYPES.index(kind))
        rows["demo"].append(e)
        if HQ is not None:
            rows["hand_q"].append(HQ[t])
bank = {k: np.asarray(v) for k, v in rows.items()}
for k in ("q", "glass", "up", "target") + (("hand_q",) if HQ is not None else ()):
    bank[k] = bank[k].astype(np.float32)
np.savez(args.out, types=np.array(TYPES), **bank)
held_ok = [(bank["held"][bank["type"] == i].mean(), bank["closed"][bank["type"] == i].mean(), int((bank["type"] == i).sum()))
           for i in range(len(TYPES))]
print(f"bank {args.out}: {len(bank['type'])} starts from {len(set(bank['demo']))} demos; per type (held, closed): "
      + ", ".join(f"{t} n {n} ({a:.2f}, {b:.2f})" for t, (a, b, n) in zip(TYPES, held_ok)))
