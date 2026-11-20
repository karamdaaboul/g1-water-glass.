#!/bin/bash
# Pick-and-place demos for the QC trainer (2026-11-19; v5 2026-11-20: friction grasp, no weld): scripted demonstrator in G1PPTaskEnv, successes only.
cd ~/workspaces/g1_water_glass/v4
source ../env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$PYTHONPATH:$PWD
$PY gen_pp_demos.py --device cuda:0 --num_envs 32 --rounds 6 --seed 0 --out demos/demos_pp_v5.npz > demos/demos_pp_v5.log 2>&1
echo "demos exit $?" > demos/demos.done
