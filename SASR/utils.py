from minigrid.wrappers import *

from RLEnvs.MyFetchRobot import reach, push
from RLEnvs.MyMujoco import ant_v4, humanoid_v4, humanoidstandup_v4, walker2d_v4

import cv2
import collections


# ============================================================
# Mario Environment Wrappers
# ============================================================

class GymToGymnasiumWrapper(gym.Wrapper):
    """Wraps an old-style gym env (4-return step) to gymnasium-style (5-return step)."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}


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
    """Convert RGB to grayscale and resize to (84, 84)."""

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, self._height, self._width), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self._width, self._height), interpolation=cv2.INTER_AREA)
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
    """Replace Mario's original reward with sparse reward:
       reward = delta_score / 1000 + flag_get_bonus (+1)
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_score = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Compute sparse reward
        current_score = info.get("score", 0)
        delta_score = current_score - self._prev_score
        self._prev_score = current_score

        sparse_reward = delta_score / 1000.0
        if info.get("flag_get", False):
            sparse_reward += 1.0

        return obs, sparse_reward, done, truncated, info


def mario_env_maker(env_id="SuperMarioBros-v0", seed=1, render=False, movement="simple"):
    """
    Create a Super Mario Bros environment with standard preprocessing.
    :param env_id: the Mario environment ID
    :param seed: the random seed
    :param render: whether to render the environment
    :param movement: action set - 'simple' (7 actions), 'right_only' (5), or 'complex' (12)
    :return: preprocessed environment with obs_space Box(4,84,84) float32 and Discrete action_space
    """
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
        env = gym_super_mario_bros.make(env_id, render_mode="human", apply_api_compatibility=True)
    else:
        env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True)

    env = JoypadSpace(env, actions)

    # Convert old gym API → gymnasium-style 5-return step
    env = GymToGymnasiumWrapper(env)
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
