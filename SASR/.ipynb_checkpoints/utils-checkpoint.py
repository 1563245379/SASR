import gymnasium as gym
import numpy as np
from minigrid.wrappers import *

from RLEnvs.MyFetchRobot import reach, push
from RLEnvs.MyMujoco import ant_v4, humanoid_v4, humanoidstandup_v4, walker2d_v4

import cv2
import collections


# ============================================================
# Mario Environment Wrappers
# ============================================================

class GymToGymnasiumWrapper(gym.Env):
    """Wraps an old-style gym env to a proper gymnasium.Env (5-return step, seed/options support)."""

    def __init__(self, old_env):
        super().__init__()
        self._old_env = old_env
        # Copy spaces (convert old gym spaces to gymnasium if needed)
        obs_sp = old_env.observation_space
        act_sp = old_env.action_space
        self.observation_space = gym.spaces.Box(
            low=obs_sp.low, high=obs_sp.high, shape=obs_sp.shape, dtype=obs_sp.dtype
        )
        if hasattr(act_sp, 'n'):
            self.action_space = gym.spaces.Discrete(act_sp.n)
        else:
            self.action_space = gym.spaces.Box(
                low=act_sp.low, high=act_sp.high, shape=act_sp.shape, dtype=act_sp.dtype
            )
        self.metadata = getattr(old_env, 'metadata', {})
        self.render_mode = getattr(old_env, 'render_mode', None)

    def step(self, action):
        result = self._old_env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, float(reward), bool(done), False, info
        return result

    def reset(self, seed=None, options=None):
        # Old gym envs don't support seed/options in reset
        result = self._old_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def render(self):
        return self._old_env.render()

    def close(self):
        return self._old_env.close()


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping), take max over last 2 frames."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """Convert RGB to grayscale and resize to (84, 84). Optionally black out top rows."""

    def __init__(self, env, width=84, height=84, blackout_top=True):
        super().__init__(env)
        self._width = width
        self._height = height
        self._blackout_top = blackout_top
        self._saved_sample = False
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, self._height, self._width), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self._width, self._height), interpolation=cv2.INTER_AREA)
        # Black out the top 15 rows to remove HUD
        if self._blackout_top:
            resized[:15, :] = 0
        if not self._saved_sample:
            cv2.imwrite("sample_observation.png", resized)
            print(f"[GrayscaleResizeWrapper] Saved sample observation to sample_observation.png")
            self._saved_sample = True
        return resized[np.newaxis, :, :]  # (1, 84, 84)


class FrameStackWrapper(gym.Wrapper):
    """Stack `n_frames` most recent frames along channel axis. Output: (n_frames, H, W)."""

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self._n_frames = n_frames
        self._frames = collections.deque(maxlen=n_frames)
        single_shape = env.observation_space.shape  # (1, H, W)
        h, w = single_shape[1], single_shape[2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_frames, h, w), dtype=np.uint8
        )

    def _get_obs(self):
        return np.concatenate(list(self._frames), axis=0)  # (n_frames, H, W)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, truncated, info


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """Normalize uint8 [0,255] observations to float32 [0,1]."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32) / 255.0


class MarioSparseRewardWrapper(gym.Wrapper):
    """Replace Mario's original reward with distance-based reward at episode end.
       reward = (x_pos - START_X) / (END_X - START_X), clipped to [0, 1].
    """

    START_X = 40
    END_X = 3160

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if done or truncated:
            x_pos = info.get("x_pos", self.START_X)
            print(x_pos)
            reward = np.clip((x_pos - self.START_X) / (self.END_X - self.START_X), 0.0, 1.0)
        else:
            reward = 0.0

        return obs, reward, done, truncated, info


class NoopResetWrapper(gym.Wrapper):
    """Execute random number of no-ops on reset (gymnasium-compatible)."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeMarioWrapper(gym.Wrapper):
    """Make loss-of-life == end-of-episode for Mario (gymnasium-compatible)."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # Detect life loss via the unwrapped Mario env
        smb = get_unwrapped_smb_env(self)
        lives = smb._life
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        smb = get_unwrapped_smb_env(self)
        self.lives = smb._life
        return obs, info


def get_unwrapped_smb_env(env):
    """Recursively unwrap to find the SuperMarioBrosEnv instance."""
    current = env
    while hasattr(current, 'env') or hasattr(current, '_old_env'):
        if hasattr(current, '_old_env'):
            current = current._old_env
        else:
            current = current.env
        if type(current).__name__ == 'SuperMarioBrosEnv':
            return current
    if type(current).__name__ == 'SuperMarioBrosEnv':
        return current
    raise ValueError("Could not find SuperMarioBrosEnv in wrapper chain")


class CurriculumMarioWrapper(gym.Wrapper):
    """Wrapper that supports curriculum learning by changing Mario's start position."""

    def __init__(self, env, curriculum_positions):
        super().__init__(env)
        self.curriculum_positions = curriculum_positions  # list of (x, y) tuples
        self._current_stage = 0
        self._smb_env = get_unwrapped_smb_env(env)

    def set_stage(self, stage_idx, render=False):
        """Set the curriculum stage, fast-forwarding Mario to the target position."""
        self._current_stage = stage_idx
        target_x = self.curriculum_positions[stage_idx][0]
        self._smb_env.set_curriculum_position(target_x, render=render)

    @property
    def current_stage(self):
        return self._current_stage

    @property
    def num_stages(self):
        return len(self.curriculum_positions)


def mario_env_maker(env_id="SuperMarioBros-v0", seed=1, render=False, movement="simple",
                    curriculum_positions=None, stuck_timeout=0):
    """
    Create a Super Mario Bros environment with standard preprocessing.
    :param env_id: the Mario environment ID
    :param seed: the random seed
    :param render: whether to render the environment
    :param movement: action set - 'simple' (7 actions), 'right_only' (5), or 'complex' (12)
    :param stuck_timeout: end episode if agent position unchanged for this many steps (0=disabled)
    :return: preprocessed environment with obs_space Box(4,84,84) float32 and Discrete action_space
    """
    import SASR.compat_patches  # noqa: F401 — apply NumPy 2.0 monkey-patches before importing nes_py/mario
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT

    if movement == "simple":
        actions = SIMPLE_MOVEMENT
    elif movement == "right_only":
        actions = RIGHT_ONLY
    elif movement == "complex":
        actions = COMPLEX_MOVEMENT
    else:
        actions = SIMPLE_MOVEMENT

    if render:
        old_env = gym_super_mario_bros.make(env_id, render_mode="human", apply_api_compatibility=True, disable_env_checker=True)
    else:
        old_env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, disable_env_checker=True)

    # Set stuck_timeout on the underlying SuperMarioBrosEnv
    unwrapped = old_env.unwrapped
    if hasattr(unwrapped, '_stuck_timeout'):
        unwrapped._stuck_timeout = stuck_timeout

    old_env = JoypadSpace(old_env, actions)

    # Convert old gym env → gymnasium env
    env = GymToGymnasiumWrapper(old_env)

    # Replace reward with sparse reward (score delta + flag_get)
    env = MarioSparseRewardWrapper(env)
    # Frame skipping: repeat action for 4 frames, take max over last 2
    env = MaxAndSkipEnv(env, skip=4)
    # Grayscale + resize to 84x84
    env = GrayscaleResizeWrapper(env)
    # Frame stacking: stack 4 most recent frames → (4, 84, 84)
    env = FrameStackWrapper(env, n_frames=4)
    # Normalize pixels to [0, 1] float32
    env = NormalizeObservationWrapper(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if curriculum_positions is not None:
        env = CurriculumMarioWrapper(env, curriculum_positions)

    return env


def dqn_mario_env_maker(env_id="SuperMarioBros-v0", seed=1, render=False, movement="complex"):
    """
    Create a Mario environment with DQN-style preprocessing (OpenAI baseline wrappers).
    Differences from SASR wrapper: NoopReset, EpisodicLife, no HUD blackout, no sparse reward.
    """
    import SASR.compat_patches  # noqa: F401
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT

    if movement == "simple":
        actions = SIMPLE_MOVEMENT
    elif movement == "right_only":
        actions = RIGHT_ONLY
    elif movement == "complex":
        actions = COMPLEX_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    if render:
        old_env = gym_super_mario_bros.make(env_id, render_mode="human", apply_api_compatibility=True, disable_env_checker=True)
    else:
        old_env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, disable_env_checker=True)

    old_env = JoypadSpace(old_env, actions)

    env = GymToGymnasiumWrapper(old_env)
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeMarioWrapper(env)
    env = GrayscaleResizeWrapper(env, blackout_top=False)
    env = FrameStackWrapper(env, n_frames=4)
    env = NormalizeObservationWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def classic_control_env_maker(env_id, seed=1, render=False):
    """
    Make the environment.
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def continuous_control_env_maker(env_id, seed=1, render=False, **kwargs):
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # + transform the reward from {-1, 0} to {0, 1}
    env = gym.wrappers.TransformReward(env, lambda reward: reward + 1.0)
    # + flatten the dict observation space to a vector
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
    )

    new_obs_length = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["achieved_goal"].shape[0]
            + env.observation_space["desired_goal"].shape[0]
    )

    # redefine the observation of the environment, make it the same size of the flattened dict observation space
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_length,), dtype=np.float32)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
