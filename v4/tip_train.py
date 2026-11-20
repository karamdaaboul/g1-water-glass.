"""Train the tipping predictor (option B, 2026-11-19) on tip_data.py pushes; evaluate on held-out pushes from another seed.

Window at step t of a push (only while the glass is still up): input = state features at t + commands; label = the glass goes
down (tilt > 45 deg or off the table) within the next H_LABEL = 15 steps (1 s). Two variants:
  chunk : state + the next 10 commands (the planned chunk, h = 10)
  action: state + the current command only (computable in the env each step: this one gives the privileged 'glass down' cost)
Metrics on held-out pushes: AUC; window-level recall / precision at thresholds; event-level catch rate = share of pushes that
went down where the risk crossed the threshold at least 3 steps (0.2 s) before the glass went down; false-alarm rate on pushes
where the glass stayed up; calibration (mean predicted vs observed by bin). Saves tipdata/tip_model_<variant>.pt.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from tip_features import tip_features

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--train", default="tipdata/train_s0.npz")
parser.add_argument("--holdout", default="tipdata/holdout_s1.npz")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
H_CHUNK, H_LABEL, LEAD = 10, 15, 3


def windows(path, variant):
    z = np.load(path)
    cur = {k: torch.tensor(z[k]) for k in ("gpos", "gquat", "bpos", "q", "closed")}
    prev = {k: torch.cat([v[:, :1], v[:, :-1]], 1) for k, v in cur.items()}  # t = 0: the previous state = the current (at rest)
    F = tip_features(cur, prev, float(z["dt"]), float(z["table_top"])).numpy()
    A, D = z["action"], z["down"].astype(bool)
    n, T, _ = F.shape
    X, Y, E, S = [], [], [], []
    for t in range(T - max(H_CHUNK, H_LABEL)):
        alive = ~D[:, : t + 1].any(1)  # glass still up at t
        y = D[:, t + 1 : t + 1 + H_LABEL].any(1)
        a = A[:, t : t + H_CHUNK].reshape(n, -1) if variant == "chunk" else A[:, t]
        X.append(np.concatenate([F[:, t], a], 1)[alive])
        Y.append(y[alive])
        E.append(np.arange(n)[alive])
        S.append(np.full(alive.sum(), t))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32), np.concatenate(E), np.concatenate(S), D


def auc(p, y):
    o = np.argsort(p)
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


report = {}
for variant in ("chunk", "action"):
    Xtr, Ytr, _, _, _ = windows(args.train, variant)
    Xte, Yte, Ete, Ste, Dte = windows(args.holdout, variant)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    dev = args.device
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    xt = torch.tensor((Xtr - mean) / std, device=dev); yt = torch.tensor(Ytr, device=dev)
    xe = torch.tensor((Xte - mean) / std, device=dev)
    lossf = nn.BCEWithLogitsLoss()  # unweighted: the output is used as a probability (cost), so keep it calibrated
    for ep in range(args.epochs):
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 4096):
            j = perm[i : i + 4096]
            loss = lossf(net(xt[j])[:, 0], yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        for gq in opt.param_groups:
            gq["lr"] = 1e-3 * 0.5 * (1 + np.cos(np.pi * (ep + 1) / args.epochs))
    with torch.no_grad():
        pe = torch.sigmoid(net(xe)[:, 0]).cpu().numpy()
    rep = {"train_windows": int(len(Ytr)), "train_pos": float(Ytr.mean()), "holdout_windows": int(len(Yte)), "holdout_pos": float(Yte.mean()),
           "auc": auc(pe, Yte), "brier": float(((pe - Yte) ** 2).mean())}
    # event level: per held-out push
    n_push = Dte.shape[0]
    first = np.where(Dte.any(1), Dte.argmax(1), -1)
    for thr in (0.1, 0.2, 0.3, 0.5):
        flag = pe >= thr
        tp = (flag & (Yte > 0)).sum(); rep[f"thr{thr}"] = {"window_recall": float(tp / max(Yte.sum(), 1)), "window_precision": float(tp / max(flag.sum(), 1))}
        caught, fell, alarms, stayed = 0, 0, 0, 0
        for e in range(n_push):
            m = Ete == e
            if first[e] >= 0:
                fell += 1
                early = m & (Ste <= first[e] - LEAD)
                caught += int((flag & early).any())
            else:
                stayed += 1
                alarms += int((flag & m).any())
        rep[f"thr{thr}"].update({"event_catch": caught / max(fell, 1), "event_miss": 1 - caught / max(fell, 1), "false_alarm_pushes": alarms / max(stayed, 1)})
    bins = np.linspace(0, 1, 6)
    rep["calibration"] = [(f"{bins[i]:.1f}-{bins[i + 1]:.1f}", int(((pe >= bins[i]) & (pe < bins[i + 1] + (i == 4))).sum()),
                           round(float(Yte[(pe >= bins[i]) & (pe < bins[i + 1] + (i == 4))].mean()) if ((pe >= bins[i]) & (pe < bins[i + 1] + (i == 4))).any() else -1, 3))
                          for i in range(5)]
    rep["pushes_down"] = int((first >= 0).sum()); rep["pushes"] = int(n_push)
    report[variant] = rep
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std, "variant": variant, "h_chunk": H_CHUNK, "h_label": H_LABEL,
                "in_dim": int(Xtr.shape[1]), "report": rep}, f"tipdata/tip_model_{variant}.pt")
    print(f"MODEL {variant}: {json.dumps(rep)}", flush=True)
json.dump(report, open("tipdata/tip_report.json", "w"), indent=1)
