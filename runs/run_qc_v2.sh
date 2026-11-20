#!/bin/bash
# G1 water-glass v2 (glass in the hand) training (2026-11-18): QC trainer (local QC branch agxcave-spline-g1, worktree ~/workspaces/qc_upstream-g1) on the
# Isaac Lab env server g1_water_server.py. Usage: run_qc.sh VARIANT SEED STEPS
#   ours   = the paper's final method minus its excavator-only parts: spline chunks h 10, K 4, C1, hard rate limit (p95 of
#            the demos, floor 1.2 / h), spill cost critic with dual ascent (budget 0.02 = 2 % of the water expected from demo states)
#   qcrlpd = QC-RLPD baseline, free 5-step chunks, no constraint
# Common (as qc_v2f_aachen.sbatch): discount 0.995, utd 10, offline ratio 0.5, 2000 random steps, reward scale 10, 16 envs.
set -euo pipefail
VARIANT=$1; SEED=$2; STEPS=$3
QC=/home/human/workspaces/qc_upstream-g1
G=/home/human/workspaces/g1_water_glass; V=$G/v2
case "$VARIANT" in
  ours) VFLAGS="--horizon_length 10 --spline_points 4 --spline_c1 --hard_rate --hard_pct 95 --contact_budget 0.02" ;;
  qcrlpd) VFLAGS="--horizon_length 5" ;;
  *) echo "unknown VARIANT $VARIANT"; exit 2 ;;
esac
O=$G/runs_v2/${VARIANT}_s${SEED}
mkdir -p "$O"
# PCI order: device 0 = the RTX PRO 4500 Blackwell (CUDA's default order puts the RTX 4000 Ada first, which another job uses)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=$QC:/home/human/venvs/qc_pylib
cd $QC
echo "$(date -Is) start $VARIANT seed $SEED steps $STEPS qc $(git rev-parse --short HEAD)" >> $O/progress.log
/home/human/venvs/agx_plain/bin/python -u train_qc_agxcave.py --task isaaclab:$V/g1_inhand_server.py --demos $V/demos_v2.npz \
  --num_envs 16 --seed "$SEED" --offline_steps 0 --online_steps "$STEPS" $VFLAGS --discount 0.995 --utd_ratio 10 \
  --offline_ratio 0.5 --start_training 2000 --reward_scale 10 --log_interval 2000 --save_interval 50000 \
  --agent=agents/qcrlpd.py --run_root "$O/run" > "$O/train.log" 2>&1 && RC=0 || RC=$?
echo "$(date -Is) train exit $RC" >> $O/progress.log
