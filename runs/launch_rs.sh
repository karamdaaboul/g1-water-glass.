#!/bin/bash
# G1 with our method's reset states (2026-11-20, user: "the used method should be the same ... reset state, success-guided or
# uniform"): every reset drawn from start_bank_v6's demo starts + as many standard starts, type_pooled success-guided
# (G1P_RESET_MODE=adaptive). Otherwise the _rec runs' setup (demos v5, budgets, friction grasp). Usage: launch_rs.sh SEED GPU [WAIT_FILE WAIT_PATTERN]
S=$1; GPU=$2; WF=${3:-}; WP=${4:-}
G=/home/human/workspaces/g1_water_glass
if [ -n "$WF" ]; then until grep -q "$WP" "$WF" 2>/dev/null; do sleep 60; done; fi
cd $G
G1P_RESET_STATES=$G/v4/demos/start_bank_v6.npz G1P_RESET_MODE=adaptive DEMOS=$G/v4/demos/demos_pp_v5.npz BUDGETS=0.01,r1.0,r1.0,- \
  GPU=$GPU runs/run_qc_pp.sh ours_v5 $S 500000 _rs
