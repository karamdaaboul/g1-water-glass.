#!/bin/bash
# G1 with physics randomisation (G1P_DR=1: glass mass and fill, finger/glass friction, gains, command delay, table height) and our
# method's reset states (2026-11-20; user: "when it's trained add physics randomisation" + "the used method should be the same"):
# demos v6_dr, reset-state bank v7_dr, success-guided. Usage: launch_rsdr.sh SEED GPU [WAIT_FILE WAIT_PATTERN]
S=$1; GPU=$2; WF=${3:-}; WP=${4:-}
G=/home/human/workspaces/g1_water_glass
if [ -n "$WF" ]; then until grep -q "$WP" "$WF" 2>/dev/null; do sleep 60; done; fi
cd $G
G1P_DR=1 G1P_RESET_STATES=$G/v4/demos/start_bank_v7_dr.npz G1P_RESET_MODE=adaptive DEMOS=$G/v4/demos/demos_pp_v6_dr.npz \
  BUDGETS=0.01,r1.0,r1.0,- GPU=$GPU runs/run_qc_pp.sh ours_v5 $S 500000 _rsdr
