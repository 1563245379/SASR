# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Reference implementation of **SASR** (Self-Adaptive Success Rate reward shaping, ICLR 2025), extended in this fork with a **PPO** baseline (continuous control + discrete/Super Mario Bros) for a COMP579 course project. The SASR algorithm is SAC with an additional Beta-distributed shaped reward derived from a KDE over success/failure trajectory states. PPO is a single-environment CleanRL-style implementation kept deliberately separate from SASR so the two can be compared on the same env wrappers.

## Setup

```bash
pip install -r requirements.txt
```

The repo has been tested with `pytorch==2.0.1+cu117`. The Mario stack (`gym_super_mario_bros`, `nes_py`) ships uint8 arithmetic that overflows under NumPy 2.0; [SASR/compat_patches.py](SASR/compat_patches.py) monkey-patches `nes_py._rom.ROM` and `gym_super_mario_bros.smb_env.SuperMarioBrosEnv` at import time. **It must be imported before `gym_super_mario_bros`/`nes_py`** — `mario_env_maker` already does this; any new entry point that touches Mario must do the same.

There is no test suite, linter config, or build step.

## Common commands

Train (each script runs a single seed; logs to `./runs/<run-name>/` for TensorBoard, saves to `--save-folder`):

```bash
# SASR on continuous control (defaults to MountainCarContinuous-v0)
python run-SASR.py --env-id MyMujoco/Ant-Height-Sparse --seed 1

# PPO on continuous control. Defaults are biased toward sparse classic control
# (ent_coef=0.04, gae_lambda=0.97, hidden=256x3, log_std_init=0.3, total=2M).
python run-PPO.py --env-id MountainCarContinuous-v0 --seed 1
python run-PPO.py --num-seeds 3   # sequential seeds: seed, seed+1, seed+2

# Mario (image obs, discrete actions)
python run-SASR-mario.py --env-id SuperMarioBros-v0 --movement simple
python run-PPO-mario.py  --env-id SuperMarioBros-1-1-v1 --movement simple
```

Evaluate (loads checkpoints written by the matching `run-*.py`):

```bash
python eval-SASR.py       --env-id <id> --exp-name sasr        --indicator final --seed 1
python eval-PPO.py        --env-id <id> --exp-name ppo         --indicator final --seed 1 \
                          --hidden-dim 256 --num-hidden-layers 3
python eval-SASR-mario.py --env-id SuperMarioBros-v0       --exp-name sasr-mario --indicator final --seed 1
python eval-PPO-mario.py  --env-id SuperMarioBros-1-1-v1   --exp-name ppo-mario  --indicator final --seed 1 \
                          --reward dense   # or --reward sparse
```

TensorBoard:

```bash
tensorboard --logdir runs
```

## Architecture

### Algorithm / network split

[SASR/](SASR/) holds all algorithm code, parameterized by an env and a network class so the same algorithm class works for vector and image obs:

- [SASR/SASRAlgo.py](SASR/SASRAlgo.py) — SASR over SAC, continuous actions. The shaped reward is `Beta(α, β).sample()` where α/β are pseudo-counts derived from KDE density over success vs. failure state buffers (`KDE_RFF_sample` is the fast Random Fourier Features path; `KDE_sample` is the exact O(N²) fallback used when `--rff-dim` is `None`). Trajectories get split into success vs. failure buffers based on whether a positive reward was seen during the episode.
- [SASR/SASRAlgoDiscrete.py](SASR/SASRAlgoDiscrete.py) — SASR-Discrete (Mario), CNN backbone.
- [SASR/PPOAlgo.py](SASR/PPOAlgo.py), [SASR/PPOAlgoDiscrete.py](SASR/PPOAlgoDiscrete.py) — PPO (CleanRL-style: GAE-Lambda, clipped surrogate, advantage normalization, optional LR anneal). Single-env rollout, no vectorized envs.
- [SASR/Networks.py](SASR/Networks.py) — `SACActor`/`QNetworkContinuousControl` (vector obs), `SACActorDiscrete`/`QNetworkDiscrete` (CNN), `PPOActorCriticContinuous` (configurable depth/width, tanh-squashed Gaussian matching SAC's pre-tanh space), `PPOActorCriticDiscrete` (Categorical over the JoypadSpace action set). `CNNFeatureExtractor` (Atari-style 8/4-2-1 conv + 512 fc) is shared across the CNN heads.

### Env registration & dispatch

[RLEnvs/](RLEnvs/) defines the **sparse-reward variants** of standard Mujoco/FetchRobot tasks. Each module calls `gymnasium.envs.registration.register` at import time, so importing the package is what makes the env IDs visible to `gym.make`. [SASR/utils.py](SASR/utils.py) does this implicitly (`from RLEnvs.MyMujoco import ant_v4, humanoid_v4, ...`) — anything that uses these env IDs must import from `SASR.utils` (or otherwise import the `RLEnvs.*` modules) **before** calling `gym.make`. Env IDs in this group all start with `My...` (`MyMujoco/...`, `MyFetchRobot/...`).

Both `run-SASR.py` and `run-PPO.py` route by env-id prefix: `args.env_id.startswith("My")` → `continuous_control_env_maker` (flattens dict obs and applies `TransformReward(r → r + 1)` so {-1, 0} sparse rewards become {0, 1}). Otherwise → `classic_control_env_maker` (plain `gym.make` + optional `--reward-scale`/`--reward-offset` shaping wrapper, **classic envs only**). Adding a new sparse env means: register it under `RLEnvs/`, then start its ID with `My` so the existing dispatch picks it up.

### Mario preprocessing pipeline

`mario_env_maker` in [SASR/utils.py](SASR/utils.py) chains, in order: legacy gym→gymnasium adapter, `MarioSparseRewardWrapper` (replaces native reward with a single end-of-episode reward = clipped `(x_pos - START_X) / (END_X - START_X)`, zero per step), 4-frame skip-with-max, RGB→grayscale 84×84, 4-frame stack along channel axis (final shape `(4, 84, 84)`), uint8→float32 `[0, 1]` normalize, `RecordEpisodeStatistics`. Action set is selected by `--movement` ∈ {`simple` (7), `right_only` (5), `complex` (12)}.

If you switch the Mario reward back to dense or change the wrapper order, re-run training — the saved checkpoints encode their reward regime in their naming convention only by hand (see next section).

### Checkpoint naming gotcha (Mario PPO)

`run-PPO-mario.py`/`PPOAlgoDiscrete.save` writes `ppo-ac-{exp}-{indicator}-{seed}.pth`, but `eval-PPO-mario.py` loads `ppo-ac-{exp}-{indicator}-{seed}-{reward}.pth` (extra `--reward dense|sparse` suffix). The checked-in samples in [ppo-mario/](ppo-mario/) (`...-final-1-dense.pth`, `...-final-1-sparse.pth`) were renamed manually after training. Either rename the saved file post-hoc to add the `-dense`/`-sparse` suffix, or adjust the loader if you don't care about distinguishing reward regimes.

### Logging / output layout

- TensorBoard runs: `./runs/{exp_name}-{env_id}-{seed}-{timestamp}/`
- Checkpoints (default save folders): `./sasr/`, `./ppo/`, `./sasr-mario/`, `./ppo-mario/`. SASR also dumps the replay-buffer observations as `RB-*.npy` next to the actor/critic `.pth` files.
- `.gitignore` excludes `ppo/`, `ppo-mario/`, `runs/`, `*.pth`, `eval_results/`, etc. The sample SASR checkpoints in [SASR/](SASR/) and [sasr-mario/](sasr-mario/) are committed deliberately as references; the matching `*.pth` rule is overridden for those paths via the directory layout.
