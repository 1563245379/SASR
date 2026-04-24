# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Reference implementation of **SASR** (Self-Adaptive Success Rate reward shaping, ICLR 2025), plus a Super Mario Bros extension that adapts the algorithm to discrete, image-based control and adds curriculum learning. The repo is both a paper codebase (continuous-control sparse-reward benchmarks) and a self-contained Mario training harness with a DQN baseline.

## Install & run

```bash
pip3 install -r requirements.txt
# Plus: the vendored ./gym-super-mario-bros package (editable install from its own setup.py) for the Mario tasks.
```

- **Continuous control (paper experiments):** `python run-SASR.py --env-id <id>` — see README.md for the full env-id list. Custom sparse-reward envs are registered on import via `RLEnvs/MyMujoco/*.py` and `RLEnvs/MyFetchRobot/*.py`. `MyFetchRobot/*` and `MyMujoco/*` ids go through `continuous_control_env_maker` (flattens dict obs, shifts reward `{-1,0} -> {0,1}`); non-`My*` ids go through `classic_control_env_maker`.
- **Mario SASR:** `python run-SASR-mario.py --env-id SuperMarioBros-1-1-v1 [--curriculum] [--movement simple|right_only|complex]`. With `--curriculum`, training walks through `CURRICULUM_POSITIONS` (hardcoded in run-SASR-mario.py, goal-first to start-first) via `SASRDiscrete.curriculum_learn`.
- **DQN baseline:** `python run-dqn-mario.py` — writes to the same `runs/` + `sasr-mario/` dirs as SASR using the `{actor|qf_1|qf_2}-{exp_name}-{indicator}-{seed}.pth` naming scheme so eval scripts can load it.
- **Evaluate:** `python eval-SASR.py --env-id <id>` / `python eval-SASR-mario.py`. Loads checkpoints from `--model-dir` by convention `{actor|qf_1|qf_2}-{exp_name}-{indicator}-{seed}.pth`. `--plot-training` parses TensorBoard events under `runs/` via tbparse.
- **Manual Mario play + curriculum recording:** `python play-mario.py [--record <x>|--record-all]` writes action sequences to `curriculum_actions/target_<x>.npy`. Those `.npy` files are consumed by `SuperMarioBrosEnv.set_curriculum_position` to fast-forward the agent at stage reset.

Single-test invocation: there is no test suite — smoke-test changes by running a training script for a few thousand steps (e.g., `--total-timesteps 20000 --learning-starts 1000`) and checking the tqdm episode-return line and a run in TensorBoard.

## Architecture

The codebase has two parallel SASR implementations that share the same Thompson-sampling-on-Beta-distribution idea but differ in backbone and feature space:

- **`SASR/SASRAlgo.py` (`SASR`)** — SAC continuous-action backbone. KDE is fit directly on raw observation vectors from two persistent success/failure sample buffers (`self.success_samples`, `self.failure_samples`), optionally projected through Random Fourier Features (`rff_dim`) for linear-time density estimation. `SASR.learn` is the main training loop; `reward_weight * (p_success / (p_success + p_failure))` is added to the environment reward. `retention_rate` controls how much of the old S/F buffer survives each episode (evolving Beta prior).
- **`SASR/SASRAlgoDiscrete.py` (`SASRDiscrete`)** — SAC-Discrete + CNN backbone for `(4, 84, 84)` image obs. KDE runs on **CNN feature vectors** (default 512-dim) produced by the actor's encoder, not raw pixels. Because the encoder drifts during training, `_rebuild_buffer_features` re-extracts features for the whole replay buffer every `feature_refresh_interval` steps. Success/failure for Mario is detected via `info['flag_get']` (success) vs. death/timeout (failure). Exposes an extra `curriculum_learn(...)` loop on top of the standard `learn(...)`.

Networks live in `SASR/Networks.py`:
- Continuous: `SACActor` (tanh-squashed Gaussian), `QNetworkContinuousControl` (MLP `Q(s,a) -> scalar`).
- Discrete: `CNNFeatureExtractor` (Nature-DQN-ish CNN, output dim 512) shared pattern; `SACActorDiscrete` (categorical, exposes `get_features` for KDE), `QNetworkDiscrete` (`Q(s) -> |A|` vector).

Environment construction (`SASR/utils.py`) is the glue layer and contains most of the Mario-specific code:
- `continuous_control_env_maker` / `classic_control_env_maker` — see above.
- `mario_env_maker` — full preprocessing stack in this order: `JoypadSpace` → `GymToGymnasiumWrapper` (old-gym ↔ gymnasium 5-tuple bridge) → `MarioSparseRewardWrapper` (replaces dense reward with a terminal `(x_pos - 40) / (3160 - 40)` signal) → `MaxAndSkipEnv(skip=4)` → `GrayscaleResizeWrapper` (84×84, HUD blackout top-15 rows) → `FrameStackWrapper(4)` → `NormalizeObservationWrapper` (uint8 → float32 [0,1]) → `RecordEpisodeStatistics` → optional `CurriculumMarioWrapper`. Grayscale wrapper also writes `sample_observation.png` on first step — harmless but unexpected if you grep for stray PNGs.
- `dqn_mario_env_maker` — same backbone but uses DQN-style preprocessing (NoopReset, EpisodicLife, no HUD blackout, no sparse reward) to match the dueling-DQN baseline exactly.
- `get_unwrapped_smb_env(env)` walks the wrapper chain across **both** gymnasium `.env` and our custom `._old_env` link to reach the underlying `SuperMarioBrosEnv`; use this helper whenever you need to touch raw Mario state.

The vendored `gym-super-mario-bros/` is a local fork, not pristine. Custom additions in `smb_env.py` that the rest of the repo depends on:
- `set_curriculum_position(target_x, ...)` — fast-forwards by replaying an action sequence loaded from `curriculum_actions/target_<x>.npy`. Recording these sequences is the job of `play-mario.py`.
- `_stuck_timeout` / `_is_stuck` — terminates the episode when `x_pos` stagnates for N steps; surfaced through `mario_env_maker(stuck_timeout=...)`.
- `info['flag_get']`, `info['x_pos']` — used by `MarioSparseRewardWrapper` and by SASR success/failure classification.

## Cross-cutting conventions

- **NumPy 2.0 compat:** `SASR/compat_patches.py` monkey-patches `nes_py._rom.ROM` and `gym_super_mario_bros.smb_env.SuperMarioBrosEnv` to cast `uint8` RAM reads to `int` before arithmetic, and restores `np.bool8`. Anything that imports Mario code **must** do `import SASR.compat_patches` before `import gym_super_mario_bros` — both env makers and `play-mario.py` already do this; new entry points need to follow suit.
- **Checkpoint naming:** `{actor|qf_1|qf_2}-{exp_name}-{indicator}-{seed}.pth`, saved to `--save-folder`. Eval scripts and the DQN baseline both follow this. When adding a new training entry point, keep the naming stable so eval works.
- **TensorBoard logs:** every run writes to `./runs/{exp_name}-{env_id_or_'mario'}-{seed}-{timestamp}/`. The `runs/` dir is the single source of truth for learning curves; `--plot-training` in eval scripts reads it.
- **Seeds:** algorithms set `random`, `numpy`, `torch` seeds in `__init__` and expect the caller to pass `--seed`. Determinism also depends on `torch.backends.cudnn.deterministic = True` which is already set.
