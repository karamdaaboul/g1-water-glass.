# G1 water-glass task (Isaac Lab + Newton) and our method on it

**Current version: v4 (`v4/`), pick-and-place of a NORMAL drinking glass of water (6 cm x 12 cm, held around its body by the
Dex3, as Unitree's Isaac-PickPlace-Cylinder-G129-Dex3 task) with Warp water.** The wine-glass variant is `v4/g1_pp_env_wine.py`.
Robot = G1 29-DoF + Dex3-1, the same model type as unitreerobotics/unitree_sim_isaaclab (`g129` + Dex3), but Isaac Lab's
G1_29DOF_CFG asset (Unitree's USDs come from their Hugging Face dataset) and Newton instead of Isaac Sim 6.2 / PhysX.
Normal glass: 6 PBF iterations (3 let the tall column bounce: demo spill 5-9 %); leak fix (particles inside the cavity are only
projected back inwards); validation 1.34 cm free: tilt spill from ~30 deg, 2/3/4 m/s^2 -> 2.0/3.8/5.8 %; demo 8/8, spill mean 0.5 %.
Video `videos/g1_pick_place_glass.mp4`.
2026-11-19 user: the hand passed through the glass (robot collisions were off). Fixed: robot collisions ON, glass = solid
cylinder collider, weld only for holding; ready pose with the hand above the glasses; ring centre (0.08, 0.045); scripted path
up -> over -> down at a pre-grasp point 12 cm on the hand's open side -> slide in -> close -> ... -> back out through the open side.
Check (8 envs): glass pushed 1.2 mm before the grasp (max 2.5), 8/8 placed upright, spill mean 0.23 %.
Video `videos/g1_pick_place_glass_contact.mp4`. The no-contact version is kept as g1_pp_env_nocollide.py / pp_demo_nocollide.py.
Earlier text: pick-and-place of a full wine glass with Warp water. v2 (glass in the hand, below) is kept;
the user asked for a harder, pick-and-place task on 2026-11-19.

## v3/v4 water: Warp position-based fluid (`v3/pbf_water.py`)
- Newton's implicit MPM water at RL speed (1 cm grid, 120-480 Hz) spilled only past ~20 deg where real water spills at ~5 deg,
  even in a frictionless shallow coupe (`v3/tilt_check.py`); Isaac Lab's realistic water demo needs 3 mm / 800 Hz; Isaac Lab's
  RL pouring task (Franka Pour) uses a granular medium. Newton has no other fluid solver.
- PBF in Warp: unilateral density constraint, XSPH, glass as an analytic moving solid, one-way; resets from a pre-settled state.
  Validated (`v3/pbf_validate.py`, 8 cm bowl, 1 cm free): level 5.99 cm (target 6.00); tilt 20/25 deg -> 1.1/7.7 % spill; lateral
  2/3/4 m/s^2 -> 2.9/9.8/17.7 %; 247 env-steps/s for 64 glasses (2,339 particles each).
- In the robot env (`v3/g1_wine_pbf_env.py`, glass held by the stem): smooth upright carries spill <= 0.5 %, tilts >= 20 deg spill.

## v4: pick and place (`v4/g1_pp_env.py`, `v4/pp_demo.py`)
- Table (top 0.75 m), free wine glass (foot collides with the table; bowl/stem visual), random start and target spots >= 12 cm apart.
- Grasp: per-world MuJoCo weld palm-glass added to the Newton builder at MODEL_INIT (a prestartup event registers the callback;
  Isaac Lab drops USD joints between two assets); switched on with the current relative pose when the hand closes, off on release.
  Robot collisions are off (visual fingers): the Dex3 friction grasp failed in Newton (v2 probes).
- Scripted demonstrator: 8/8 placed (0.1-2.0 cm, upright), spill 0.1-3.7 % (mean 1.4 %). Video `videos/g1_pick_place_demo.mp4`.
- Next (after the user's review): task (observations, success = placed + released + spill <= threshold), demos, train ours.

**v2 (`v2/`): the glass is held IN the right Dex3 hand.** v1 (below) welded the glass beside the palm; the user
rejected it on 2026-11-18 ("the G1 should carry the glass in the hand"). v1's runs were stopped; its results are not used.

## v2: glass in the hand (`v2/g1_inhand_env.py`, `v2/g1_inhand_task.py`)
- Scene as Isaac Lab's fixed-base G1 manipulation env (IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs: G1_29DOF_CFG + Dex3,
  pelvis fixed). That env itself needs pink / pinocchio / isaacteleop (not installed), so only its scene is reused.
- Round glass, 5 cm outside, 6 mm walls, 8 cm deep, ~50 ml of water, welded INSIDE the ring of thumb + index + middle finger,
  fingers closed around it in Isaac Lab's Dex3 gripper pose (tri-hand retargeter: 1.0/1.0, thumb 0/-0.4/-0.7).
- A friction-only grasp was tried first and failed: closing the hand pushes the glass out (`v2/probe_grasp.py`), placements
  touching the finger meshes give NaN contacts (`v2/probe_place.py`).
- Home: palm z axis vertical (`v2/find_home.py`), glass upright at (0.30, -0.12, 0.81). Target bank `v2/target_bank.pt`: 1,349
  upright poses (`v2/gen_target_bank.py`, kinematics only). Task, rewards, observations and all water fixes as v1.
- Checks: still 0 % spill, 0.03 deg; scripted carry fwd/left/up/back 0 % spill, <= 0.05 deg (`videos/g1_inhand_grip.mp4`).
- Demos `v2/demos_v2.npz`: 160/160 attempts succeeded, mean final spill 0.04 %, 5,071 transitions
  (`videos/g1_inhand_demos_60-67.mp4`). Demonstrator on the fixed eval targets: 128/128, spill 0.04 %, 2.18 s, tilt 0.13 deg.
- Training: `runs/run_qc_v2.sh ours|qcrlpd SEED STEPS` -> `runs_v2/`; server `v2/g1_inhand_server.py`.

# v1 (superseded): glass welded beside the palm

Started 2026-11-18. A fixed-base Unitree G1 carries a glass of water (MPM fluid) with its right arm to a target without spilling.
Isaac Lab main (`/tmp/agxcave-isaaclab`, source on PYTHONPATH via `env.sh`) + Newton 1.9 (MuJoCo-Warp for the robot,
implicit MPM for the water, coupled one way), venv `~/workspaces/agxcave/.venv-isaaclab-gpu`. No new venv was created.

## Task (`g1_water_task.py`)
- Robot: G1 29-DoF + Dex3 hands, pelvis fixed at 0.8 m. Policy at 15 Hz (sim 120 Hz, decimation 8); actions = 7 right-arm joint
  velocity commands in [-1, 1] x 1 rad/s, integrated into position-servo targets (as AGXCave drives the excavator).
- Glass: 6 x 6 cm inside, 9 cm deep, 12 mm walls, 15 mm bottom, 150 g, one watertight mesh welded to the palm.
  Water: 5 cm (180 ml), 1,210 MPM particles, 1 cm voxels.
- Episode: start at home (glass upright, water at rest); target = a glass position from `target_bank.pt` (2,572 reachable
  upright poses), at least 12 cm from home. 8 s limit.
- Success (reward 1, terminate): glass within 4 cm of the target, slower than 0.1 m/s, tilted < 15 deg, at most 10 % spilled.
  Failure: more than 30 % spilled, or the glass leaves the safe box. Cost per step: increase of the spilled fraction.
- Observation (37): arm q, arm qd, glass position, glass up vector, glass velocity, target offset, water in glass + its centre
  of mass, last action.

## Physics problems found and fixed (all in `g1_water_env.py`, `g1_glass_env.py`)
1. Reset teleport blasted the water out: Isaac Lab runs forward kinematics only at the next physics step, after the MPM
   history reset cached the pre-reset glass pose, so the next collider velocity was (jump / dt). Fix: FK before the MPM reset.
   Plus one held step at the start of each episode.
2. Water leaked through the walls when moving sideways: `collider_velocity_mode` forward (velocity not matching the pose
   change). Fix: `finite_difference`, with **one** MPM substep (with 4, the first substep saw 4x the glass velocity and the rest
   0, an impulsive kick each step).
3. Box-assembled glass leaked at the corners; walls thinner than a voxel leaked. Fix: one watertight mesh, 12 mm walls.
4. Torso rang (waist damping ratio ~0.03 in G1_29DOF_CFG) and shook the glass. Fix: damped drives (arms kp 3000 / kd 100,
   waist 5000 / 300, armature 0.03).
5. Joint and body velocities read back from the coupled solver do not match the motion. The observation, success test and
   evaluation use step-mean velocities from positions instead (the water itself uses pose differences, unaffected).
6. B2 / pic512 MPM bases ran out of memory at 64 envs; Q1 / pic27 / P0 leak nothing with fixes 1-3.
Checks after the fixes: 0.15 m moves in six directions spill 0 %; a gentle shake 0 %; a violent shake sloshes out gradually
(13 % in 0.7 s); holding 0 %. Throughput 310 / 427 / 475 env-steps/s at 64 / 128 / 256 envs.

## Demos (`gen_demos.py`)
Resolved-rate demonstrator: min-jerk glass path over 2.5-4 s + tilt correction + null-space pull, damped pseudo-inverse of the
glass Jacobian. `demos_v1.npz`: 160 demos, 161/161 attempts succeeded, mean final spill 0.06 %, 5,175 transitions.
Video: `videos/g1_water_demos_60-67.mp4` (demos 60-67 replayed open loop; wide view + top view into the glass).

## Training our method
Env server `g1_water_server.py` (Isaac Lab venv) + client `isaaclab_env.py` in the QC worktree `~/workspaces/qc_upstream-g1`
(LOCAL branch `agxcave-spline-g1`, not pushed). `runs/run_qc.sh VARIANT SEED STEPS`:
- `ours`: spline chunks h 10, K 4, C1; hard rate limit (p95 of the demos, floor 0.12 per step); spill cost critic with dual
  ascent, budget 0.02 (via `--contact_budget`). The excavator-only parts (tip speed, soil slew, release cost, sampler) are off.
- `qcrlpd`: the QC-RLPD baseline (free 5-step chunks, no constraint).
Common: 16 envs, utd 10, offline ratio 0.5, 2,000 random steps, reward scale 10, discount 0.995. GPU: the RTX PRO 4500
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`, device 0); CUDA's default order makes `cuda:0` the RTX 4000 Ada.

## Evaluation
`eval_qc_g1.py --load_run <run dir> --checkpoint N --out eval/<name>.json` (QC worktree, agx_plain + `~/venvs/qc_pylib`), and
`G1W_EVAL_TARGETS=7 gen_demos.py --num_envs 32 --eval_json eval/demonstrator.json` for the demonstrator: 32 envs x 4 episodes on
fixed targets, deterministic actor.

## Results
(see below, filled in when the runs finish)
