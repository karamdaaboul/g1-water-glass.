# Isaac Lab main (/tmp/agxcave-isaaclab) source packages on PYTHONPATH for .venv-isaaclab-gpu (no installs).
S=/tmp/agxcave-isaaclab/source
export PYTHONPATH=$S/isaaclab_tasks:$S/isaaclab_rl:$S/isaaclab_assets:$S/isaaclab_contrib:$S/isaaclab_physx:$S/isaaclab_newton:$S/isaaclab_visualizers:/home/human/workspaces/g1_water_glass:${PYTHONPATH:-}
PY=/home/human/workspaces/agxcave/.venv-isaaclab-gpu/bin/python
