#!/bin/bash
# Friction-grasp sweep (2026-11-20): run.sh GPU "TAG:ENV=VAL,ENV=VAL" ...
cd /home/human/workspaces/g1_water_glass/v4
source ../env.sh >/dev/null 2>&1
GPU=$1; shift
for spec in "$@"; do
  tag=${spec%%:*}; vars=${spec#*:}
  ( export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU G1P_ACT_DELAY_P=0 G1P_OBS_NOISE=0
    IFS=','; for kv in $vars; do [ -n "$kv" ] && export "$kv"; done; unset IFS
    timeout 1500 $PY friction_grasp_test.py --num_envs 16 --device cuda:0 > grasp_sweep/$tag.log 2>&1 )
  echo "$(date -Is) $tag exit $? | $(grep -h 'FRICTION GRASP\|CLOSED-HAND\|^SLIP' grasp_sweep/$tag.log | tr '\n' ' ')" >> grasp_sweep/summary.txt
done
