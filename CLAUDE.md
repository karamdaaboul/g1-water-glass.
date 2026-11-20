# CLAUDE.md — G1 water-glass environment

Guidance for Claude Code sessions that work on **the environment** (scene, physics, task, demonstrator, start bank, evaluation
layouts). The **algorithm** (our QC method: spline action chunks, hard limits, cost critics) lives in another repository and is
worked on in a separate session; see "Interface with the algorithm" before changing anything the policy sees.

## What this is

A fixed-base Unitree **G1 29-DoF + Dex3-1** (right hand) picks up a **full glass of water** from a table and places it on a target
spot **without spilling**, built on **Isaac Lab (main) + Newton** (MuJoCo-Warp for the rigid bodies, our own Warp PBF water).
The task exists to test the algorithm on a humanoid and to move toward a sim-to-real-ready dexterous task.

Current version: **v4 pick-and-place, friction grasp (no weld), 2026-11-20** — `v4/`.

## Repositories and folders

| Path | What it is |
|---|---|
| this repo `karamdaaboul/g1-water-glass.` (note the trailing dot in the name: clone URL `git@github.com:karamdaaboul/g1-water-glass..git`) | environment code only |
| `~/workspaces/g1-water-glass` | this clone: **environment design work happens here** |
| `~/workspaces/g1_water_glass` | the original working copy (same repo) **with all data**: `runs_pp/` (training runs), `eval_pp/` (evaluations, traces), `videos/`, `v4/demos/*.npz` (demos, start banks). The training/eval launchers in `runs/*.sh` use absolute paths into this folder. Sync code with `git pull` there after pushing from here. |
| `karamdaaboul/qc`, branch `spline` (worktree `~/workspaces/qc_upstream-g1`) | the algorithm and the training/eval clients (`isaaclab_env.py`, `train_qc_agxcave.py --task isaaclab:<server.py>`, `eval_qc_g1pp.py`); see its `FORK.md` |

## Runtime (no new virtual environments)

- Isaac Lab + Newton venv: `~/workspaces/agxcave/.venv-isaaclab-gpu`; `source env.sh` sets `PY` and puts Isaac Lab main
  (`/tmp/agxcave-isaaclab/source/*`) and this folder on `PYTHONPATH`.
- The QC trainer/evaluator runs in `~/venvs/agx_plain` (JAX) and talks to the env server (`v4/g1_pp_server.py`, Isaac Lab venv)
  over a socket.
- GPUs: always `export CUDA_DEVICE_ORDER=PCI_BUS_ID` (device 0 = RTX PRO 4500 Blackwell 32 GB, 1 = RTX 4000 Ada 20 GB; the
  default CUDA order swaps them). Other sessions share the GPUs, RAM (62 GB, no swap) and disk (often > 95 % full: check `df`).
- Quick checks, from `v4/` after `source ../env.sh` and `export PYTHONPATH=$PYTHONPATH:$PWD`:
  - scripted demonstrator: `$PY gen_pp_demos.py --device cuda:0 --num_envs 16 --rounds 1 --out demos/smoke.npz`
  - friction grasp: `$PY friction_grasp_test.py --num_envs 16 --device cuda:0` (sweep driver `grasp_sweep/run.sh`)
  - start bank: `$PY test_start_bank.py --num_envs 16 --resets 3 --device cuda:0`

## Version history (why things are the way they are)

| Version | Folder | What changed | Outcome |
|---|---|---|---|
| v1 | root `g1_water_*.py` | carry a glass welded beside the palm; MPM water | rejected by the user: the glass must be IN the hand |
| v2 | `v2/` | glass held in the Dex3 ring (weld); friction grasp failed (closing pushed the glass out, NaN contacts) | carry task works; user wanted a harder task |
| v3 | `v3/` | water rewritten as **Warp PBF** (`v3/pbf_water.py`): MPM spilled only past ~20° tilt; PBF validated against hydrostatics and tilt/acceleration spill | PBF is the water model since |
| v4 | `v4/` | **pick-and-place** of a normal 6 × 12 cm glass from a table to a target spot; robot faces the table (+x) | current task |

v4 milestones (all 2026-11-19/20; QC method, results on 64 or 128 held-out layouts):

| Step | Change | Test success |
|---|---|---|
| first run | weld grasp | 52 % (pushed the glass instead of carrying it) |
| option 5 | push cost on the free glass | 77 % |
| option 3 | + partial-progress start bank (half the resets start from demo states) | 83 % |
| v5 | + waist (yaw, pitch) for reach, hard lift-off after release, real G1 arm torque limits, observation noise, 1-step action delay in half the episodes | **94.5 / 93.8 %** (2 seeds, weld) |
| friction | **no weld**: real friction grasp (see Physics) | **86.7 / 89.8 %** (2 seeds, best checkpoint 400k of 500k) |

## Current design (v4, `v4/g1_pp_env.py` + `v4/g1_pp_task.py`)

### Scene and physics (`g1_pp_env.py`)
- Isaac Lab `G1_29DOF_CFG` + Dex3, root fixed at 0.80 m, rot (0,0,0,1) = facing +x (the table). Self-collision off, robot
  collisions on. Sim dt 1/120 s, decimation 8 → **15 Hz policy**. MuJoCo-Warp solver `newton`, `implicitfast`, 100 iterations,
  3 substeps.
- **Contacts for grasping: `elliptic` friction cone, `impratio` 10** (`G1P_CONE`, `G1P_IMPRATIO`). The default pyramidal cone with
  impratio 1 let a squeezed glass creep out of the hand (2.2 cm per 2.5 s carry, 2 of 16 dropped); see `v4/grasp_sweep/`.
- Hand: Dex3 joints PD kp 20 / kd 2 (`G1P_HAND_KP`), **effort limit 1.4 N·m per finger joint** (Dex3-1 spec; `G1P_HAND_EFFORT`).
  Closing pose `HAND_CLOSED` scaled by `G1P_HAND_SCALE` (0.85).
- **Grip point (Dex3 ring centre) `GRIP = (0.069, 0.051)` in the palm frame** — where the closed fingers actually centre the glass;
  the old (0.08, 0.045) shoved the glass 1.2 cm while closing.
- Arms: position servo kp 3000 / kd 100; **real G1 torque limits** shoulder/elbow/wrist-roll 25 N·m, wrist pitch/yaw 5 N·m
  (`G1P_REAL_TORQUE=0` restores 300).
- Glass: 6 cm × 12 cm straight glass, **0.40 kg** (glass + water; `G1P_GLASS_MASS`), solid-cylinder collider, μ 0.8
  (`G1P_GLASS_MU`). The water is visual/one-way: it follows the glass, it does not push it.
- Water: Warp PBF (`v3/pbf_water.py`), spacing 5 mm, 16 substeps, 6 iterations, **XSPH 0.1** (`G1P_PBF_XSPH`; halves the rest
  jitter), free space above the water `G1P_FREE` 0.0084 m. Spill = fraction of particles outside the bowl.
- Table top 0.75 m. Glass start x 0.30–0.40, y −0.26…−0.06; target x 0.28–0.40, y −0.30…−0.04; at least 12 cm apart.
  Evaluation layouts are a fixed seeded bank: `G1P_EVAL_SEED=<seed>` (validation 3, test 7), env i's k-th episode = entry k·N+i.

### Task (`g1_pp_task.py`)
- **Actions (10)**: 7 right-arm joint velocity commands (×1 rad/s) + waist yaw, pitch (×0.5 rad/s), integrated into position
  targets; + 1 gripper command (> 0 close, ≤ 0 open). Latency: in `G1P_ACT_DELAY_P` (0.5) of the episodes every command arrives
  one step late.
- **Grasp** (`G1P_GRASP`, default `friction`): the fingers alone hold the glass. "held" = hand closed for ≥ `CLOSE_STEPS` (6) and
  the glass in the ring (axis within `RING_R` 2 cm of the ring centre, height within `RING_DZ` 3 cm, tilt < 10°), refreshed after
  every physics step (`_post_physics`). `G1P_GRASP=weld` restores the old palm–glass weld.
- **Release gate**: the hand may open a held glass only when it stands on the table (base within 1.5 cm, tilt < 8°).
- **Observation** (121 with the trainer's extra previous action): `[privileged 11 | constraint 6·N_J | actor 2·N_J+3+15+N_J+1]`,
  N_J = 9; layout in `obs_layout()`.
  - privileged (critics only): glass velocity, angular velocity, spill, water COM, hand–glass distance;
  - constraint (hard limits only): ring-centre and palm Jacobians over the 9 joints;
  - actor (what a real robot could measure): joint positions and velocities, hand closed, held, released, ring centre, glass base,
    glass up axis, target relative to the glass, the hand's open direction, last action.
  - Sensor noise on the actor part (`G1P_OBS_NOISE`): joints 0.002 rad, joint speeds 0.02 rad/s, glass position 3 mm (the target
    inherits it), up axis 0.005, ring 1 mm.
- **Reward**: +1 on success (sparse). **Costs** (per step, 4 channels, to the cost critics): spill increase; glass down + `PUSH_K`
  (10) × free-glass displacement; glass speed²; hand within `CONTACT_D` (2 cm) of the free glass.
- **Success**: glass within 3 cm of the target, tilt < 5°, not held, hand ≥ 4 cm away, spill ≤ 2 % (`G1P_SUCCESS_SPILL`).
  **Failures**: glass down (> 45°), spill > 30 %, time out (`G1P_EPISODE_S` 20 s = 300 steps).
- **Reset states** (our method's, the same as the excavator paper's; `G1P_RESET_STATES=<start bank npz>`, added 2026-11-20):
  every reset draws one entry from a fixed bank of standard starts (seeded layouts, robot at READY; as many as the demo starts,
  `G1P_RESET_STANDARD_FRACTION` 0.5) and demo starts (below), and the episode's success is recorded against it.
  `G1P_RESET_MODE=adaptive` (default): AlphaNIST success-guided sampling pooled per start type (`type_pooled`, eps_q 0.05,
  last 64 outcomes, 2 pseudo-outcomes at 0.5: agxcave's `V1_SGS`); `uniform`: the same bank drawn uniformly (the baseline).
  `v4/sgs_sampler.py` is agxcave's `agxcave/agxpolicy/sgs/sampler.py` byte for byte (feat/mt3d-v1 eebcde5); `v4/reset_states.py`
  builds the bank. The server log prints each start type's success estimate and draw share every 500 episodes. Off during
  evaluation (`G1P_EVAL_SEED`). Use this for every new run, so the method is the same on every task (user, 2026-11-20).
- Older runs: `G1P_START_BANK` (share `G1P_START_FRACTION` 0.5) sent half the resets to demo states drawn uniformly and the other
  half to fresh random layouts; kept only to reproduce those runs (`*_fr`, `*_rec`).
- The server (`g1_pp_server.py`) also reports, for evaluation only, the noise-free ring/glass/up state (`extras["clean"]`).

### Physics randomisation (`G1P_DR=1`, off by default; added 2026-11-20)
Drawn per environment at every reset: glass mass = glass alone 0.15–0.25 kg (`G1P_DR_GLASS_MASS`) + the water of its fill;
fill level = free space 6–25 mm (`G1P_DR_FREE`; the water is settled at the fullest level and each glass leaves out the
particles above its level, spill is measured over its own particles); right-hand finger friction 0.6–1.2 per shape
(`G1P_DR_FINGER_MU`) and glass friction 0.5–1.0 (`G1P_DR_GLASS_MU`) with the table lowered to 0.4 (`G1P_DR_TABLE_MU`; MuJoCo
uses the larger value of a contact pair); stiffness and damping of every actuator group ×0.8–1.2 (`G1P_DR_GAIN`); command delay
0–2 policy steps (`G1P_DR_DELAY_MAX`); robot root height ±3 cm (`G1P_DR_BASE_DZ`, i.e. the table height seen from the robot;
0 on start-bank resets, whose states were recorded at the default height). The critics see the drawn values (privileged block
11 → 17), so DR runs need demos and start banks recorded with `G1P_DR=1` (`demos_pp_v6_dr.npz`, `start_bank_v7_dr.npz`).

### Demo starts of the reset-state bank
`make_start_bank.py` builds pregrasp / grasped / carry / lower starts from the demos; `--recovery` adds three more start types,
slipped (closed hand over a free upright glass, not held), near_miss (lower state with the target moved 2–5 cm) and far_pregrasp
(glasses at x ≥ 0.37 twice): `start_bank_v6.npz` (1,216 starts, no DR), `start_bank_v7_dr.npz` (with DR). They are reset
states like the others; the flag name is historical.

### Demonstrator and data
- `gen_pp_demos.py`: scripted pick-and-place (9-joint damped least squares, waist weighted 0.3; approach from the open side,
  close, lift, carry, lower, release, **rise 10 cm, then retreat**). Demos record observations, actions, costs and `hand_q`.
- Current data (in `~/workspaces/g1_water_glass/v4/demos/`): `demos_pp_v5.npz` (friction grasp, 192/192 successes, 0.00 % spill)
  and `start_bank_v5.npz` (768 starts). v4 = the weld version.

## Results and evidence locations (in `~/workspaces/g1_water_glass`)
- Runs `runs_pp/ours_v5_s{0,1}` (weld) and `ours_v5_s{0,1}_fr` (friction); checkpoint selection + test by `runs/select_v6.sh`
  → `eval_pp/v5_s*_fr/{val_*,test_*}.json`.
- Friction grasp, test (128 layouts): seed 0 400k 86.7 %, seed 1 400k 89.8 %; failures mostly timeouts after a grasp; spill in
  successes 0.07–0.16 %; glass slip along the glass axis WHILE HELD 4–6 mm mean, over 3 cm in 2 episodes (seed 0, max 3.7 cm);
  measured by `eval_qc_g1pp.py` (`slip_along_max`, `lost_grip`) in `eval_pp/v5_s*_fr/test_400000_slip.json`. The slip number
  stops when the glass leaves the ring, so slip-outs are not in it: `lost_grip` (held flag drops in a closed hand) counts them
  together with flag flicker (21–39 of 128 episodes, most still successes). Seen in the video: one glass slid out of the
  fingers during the lift (tilt 8.5°) and the policy never re-grasped; one far-edge grasp closed ~2 cm too high, never held,
  and the closed hand dragged the glass along the table.
- Videos `videos/g1_pp_v5_500k*.mp4` (weld), `videos/g1_pp_fr_400k.mp4` and `_far_targets.mp4` (friction, seed 1, labelled).

## Open environment issues (the env session's backlog)
1. **"held" flickers, and real slip-outs are not separated from flicker** with the friction grasp (needs a measure that tells a
   drop in the air from a slide onto the table from a flag flicker): the ring check fails when the glass tilts past 10° or nudges out of its tolerance
   while still firmly in the hand (seen in 21–39 of 128 test episodes). Side effects: the actor's held flag flickers, the push cost
   counts the carried glass as a pushed free glass, the contact cost may fire. Needs a robust held state (hysteresis while the hand
   stays closed, or contact-based), then new demos + start bank + retraining.
2. The hand opens while still moving (release gate checks the glass, not the hand speed); the algorithm's lift-off limit only acts
   from the next plan. A "hand nearly still before opening" rule would fix both.
3. One far target (0.40, −0.05): the glass is placed ~3.1 cm off (success needs 3 cm) and the policy re-grasps instead of
   lowering again.
4. Sim-to-real gaps (from the 2026-11-20 literature review: ADEPT, Lin et al. 2025, VIRAL): Dex3 tactile sensing (33 sensors) not
   simulated (the held flag is privileged); no physics domain randomisation (glass mass/fill, friction, gains, latency range, table
   height, glass size, pushes); no calibration to the real G1/Dex3 (VIRAL: the Dex3's high gear ratio needs system ID); perception
   assumed (pose with 3 mm noise; a transparent glass is hard for depth); fixed base (real G1 sways); water never compared with
   real water.

## Interface with the algorithm (coordinate before changing)
The trained checkpoints, demos and start banks depend on these; changing one means new demos, a new start bank and retraining:
- observation term order and sizes (`obs_layout()`, `N_PRIV`, `N_CONS`, `N_ACTOR`) and the demo metadata the trainer reads
  (`hand_obs_index`, `arm_dims`, `arm_cmd_scale`, `q_default/q_lo/q_hi`, `object_height`, `contact_d`, `cost_channels`);
- the action layout (9 joint commands, gripper last) and the cost channel order;
- the server protocol (`g1_pp_server.py` ↔ `isaaclab_env.py` in the QC repo: spec, obs, step with `outcome`, `final_spill`,
  `clean`).

## Working rules (the user's)
- Ask before launching any training run and before stopping or killing any job (other sessions share the machine).
- Do not implement fixes for a question; answer with cause and options first. Do not create new virtual environments.
- Commit as `Karam Daaboul <m.k.daaboul@gmail.com>` with no Claude co-author line, and push after committing.
- Keep the user's private information out of external requests.
