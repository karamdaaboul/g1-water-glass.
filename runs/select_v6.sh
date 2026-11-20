#!/bin/bash
# Checkpoint selection + test for the G1 v5 runs (2026-11-20, user: "start with the goal to reach the best performance ever").
#   select_v6.sh SEED GPU TAG   (2026-11-20: copy of select_v5.sh for runs_pp/ours_v5_s<SEED><TAG>, e.g. TAG _fr = friction grasp)
# Waits for runs_pp/ours_v5_s<SEED> to finish, then evaluates every saved checkpoint from 100k on the VALIDATION layouts
# (eval_seed 3, 32 envs x 4 = 128 episodes), picks the best (success, then lower spill in successes, then later), and evaluates
# that checkpoint and the final one on the TEST layouts (eval_seed 7, 32 x 4 = 128 episodes; its first 2 per env are the 64
# layouts option 3 was scored on). Realism stays on as in training (torque limits, obs noise, action delay); no start bank.
set -u
S=$1; GPU=$2; TAG=${3:-}
G=/home/human/workspaces/g1_water_glass
QC=/home/human/workspaces/qc_upstream-g1
O=$G/runs_pp/ours_v5_s$S$TAG
E=$G/eval_pp/v5_s$S$TAG
mkdir -p $E
until grep -q "train exit" $O/progress.log 2>/dev/null; do sleep 60; done
RUN=$(ls -d $O/run/*/ | head -1)
unset G1P_START_BANK G1P_START_FRACTION
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=$QC:/home/human/venvs/qc_pylib
cd $QC
ev() {  # ev CHECKPOINT EVAL_SEED NAME
  [ -s $E/$3.json ] && return
  /home/human/venvs/agx_plain/bin/python eval_qc_g1pp.py --load_run $RUN --checkpoint $1 --num_envs 32 --episodes_per_env 4 \
    --eval_seed $2 --out $E/$3.json > $E/$3.log 2>&1
  mv -f $G/eval_pp/$3_server.log $E/ 2>/dev/null
  echo "$(date -Is) $3 $(python3 -c "import json; s=json.load(open('$E/$3.json'))['summary']; print(s['success'], s['outcomes'])" 2>&1)" >> $E/progress.log
}
for c in $(ls $RUN | sed -n 's/^params_\([0-9]*\)\.pkl$/\1/p' | sort -n); do
  [ "$c" -ge 100000 ] && ev $c 3 val_$c
done
BEST=$(python3 - "$E" <<'EOF'
import glob, json, re, sys
rows = []
for f in glob.glob(sys.argv[1] + "/val_*.json"):
    s = json.load(open(f))["summary"]
    rows.append((s["success"], -s.get("final_spill_success_mean", 0.0), int(re.search(r"val_(\d+)", f).group(1))))
print(max(rows)[2])
EOF
)
echo "$(date -Is) best on validation: $BEST" >> $E/progress.log
LAST=$(ls $RUN | sed -n 's/^params_\([0-9]*\)\.pkl$/\1/p' | sort -n | tail -1)
ev $BEST 7 test_$BEST
[ "$LAST" != "$BEST" ] && ev $LAST 7 test_$LAST
echo "$(date -Is) done" >> $E/progress.log
