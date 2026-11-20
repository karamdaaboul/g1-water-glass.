#!/bin/bash
# G1 pick-and-place of a water glass (v4, 2026-11-19): QC trainer (local QC branch agxcave-spline-g1, worktree
# ~/workspaces/qc_upstream-g1) on the Isaac Lab env server v4/g1_pp_server.py. Usage: run_qc_pp.sh VARIANT SEED STEPS [TAG]
#   ours = our method: spline chunks h 10, K 4, C1, hard command-change limit (p95 of the demos, floor 1.2 / h; the gripper
#          command dim 7 free), hard gentle approach (contact predicted along the chunk from robot-measurable inputs ->
#          arm commands capped at the demos' p95 near contact, x0.3 if the hand's open side does not face the glass), hard
#          effective tilt <= 10 deg while held, hard joint margin 0.05 rad; privileged critics (the actor reads the
#          observation after the privileged and constraint blocks), and three soft constraints from the env's cost channels:
#            spill -> contact critic, budget 0.01 (1 % of the water expected from the demonstrator's states)
#            down  -> cost critic, budget 0.01 (glass-down events only; the demos never drop one)
#            speed -> speed critic, budget 1.0 x the demos' own discounted mean squared glass speed
#          channel 3 (contact label) trains the approach limit's predictor.
# Common (as v2): discount 0.995, utd 10, offline ratio 0.5, reward scale 10, 16 envs; START (default 2000) random steps.
set -euo pipefail
VARIANT=$1; SEED=$2; STEPS=$3; TAG=${4:-}
START=${START:-2000}
QC=/home/human/workspaces/qc_upstream-g1
G=/home/human/workspaces/g1_water_glass; V=$G/v4
DEMOS=${DEMOS:-$V/demos/demos_pp_v3.npz}  # v3 (2026-11-19): XSPH 0.1 water; channel 1 = down + 10 x free-glass push (option 5)
BUDGETS=${BUDGETS:-0.01,0.01,r1.0,-}  # spill, disturb (down + push), speed, contact label
case "$VARIANT" in
  ours) VFLAGS="--horizon_length 10 --spline_points 4 --spline_c1 --hard_rate --hard_pct 95 --hard_rate_free_dims 7 \
                --env_cost_slots contact,cost,speed,engage --env_cost_budgets $BUDGETS \
                --approach_pct 95 --approach_mis_ratio 0.3 --hard_tilt_deg 10 --hard_joint_margin 0.05" ;;
  ours_v5) VFLAGS="--horizon_length 10 --spline_points 4 --spline_c1 --hard_rate --hard_pct 95 --hard_rate_free_dims 9 \
                --env_cost_slots contact,cost,speed,engage --env_cost_budgets $BUDGETS \
                --approach_pct 95 --approach_mis_ratio 0.3 --hard_tilt_deg 10 --hard_joint_margin 0.05 --hard_liftoff_v 0.03" ;;
    # ours_v5 (2026-11-19, user items 2 + 4): demos v4 (arm + waist commands, gripper dim 9), real G1 arm torque limits,
    # observation noise, action delay (half the episodes one step late), hard lift-off after the release (0.03 m/s horizontal)
  *) echo "unknown VARIANT $VARIANT"; exit 2 ;;
esac
O=$G/runs_pp/${VARIANT}_s${SEED}${TAG}
mkdir -p "$O"
# PCI order: device 0 = the RTX PRO 4500 Blackwell (CUDA's default order puts the RTX 4000 Ada first, which another job uses)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=$QC:/home/human/venvs/qc_pylib
cd $QC
echo "$(date -Is) start $VARIANT seed $SEED steps $STEPS start_training $START demos $DEMOS qc $(git rev-parse --short HEAD)" >> $O/progress.log
/home/human/venvs/agx_plain/bin/python -u train_qc_agxcave.py --task isaaclab:$V/g1_pp_server.py --demos "$DEMOS" \
  --num_envs 16 --seed "$SEED" --offline_steps 0 --online_steps "$STEPS" $VFLAGS --discount 0.995 --utd_ratio 10 \
  --offline_ratio 0.5 --start_training "$START" --reward_scale 10 --log_interval 2000 --save_interval 50000 \
  --agent=agents/qcrlpd.py --run_root "$O/run" > "$O/train.log" 2>&1 && RC=0 || RC=$?
echo "$(date -Is) train exit $RC" >> $O/progress.log
