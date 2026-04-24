"""
The Self-Adaptive Success Rate Shaping (SASR) algorithm for discrete action spaces,
with SAC-Discrete as the backbone. Designed for image-based environments (e.g., Super Mario Bros).

Key differences from the continuous version (SASRAlgo.py):
- SAC-Discrete: categorical policy, Q(s) -> Q-values for all actions
- CNN feature extraction for image observations
- KDE operates on CNN feature vectors (512-dim) instead of raw observations
- Success/failure detection via info['flag_get'] for Mario environments
"""

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.beta import Beta

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

import os
import random
import datetime
import time
import math

from tqdm import tqdm


class SASRDiscrete:
    """
    The Self-Adaptive Success Rate Shaping (SASR) algorithm for discrete action spaces.
    """

    def __init__(self, env, actor_class, critic_class, exp_name="sasr-mario", seed=1, cuda=0, gamma=0.99,
                 buffer_size=100000, rb_optimize_memory=False, batch_size=32, policy_lr=3e-4, q_lr=3e-4, eps=1e-8,
                 alpha_lr=3e-4, target_network_frequency=1, tau=0.005, policy_frequency=2, alpha=0.2,
                 alpha_autotune=True, reward_weight=0.6, kde_bandwidth=0.2, kde_sample_burn_in=1000, rff_dim=1000,
                 retention_rate=0.1, write_frequency=100, save_folder="./sasr-mario/",
                 feature_dim=512, feature_refresh_interval=5000, sparse_reward = True):
        """
        Initialize the SASR algorithm for discrete action spaces.
        :param env: the gymnasium-compatible environment with image observations and discrete actions
        :param actor_class: the actor class (SACActorDiscrete)
        :param critic_class: the critic class (QNetworkDiscrete)
        :param feature_dim: dimension of CNN feature vectors used for KDE (default: 512)
        """
        self.sparse_reward = sparse_reward
        self.exp_name = exp_name

        # set the random seeds
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.env = env
        self.num_actions = env.action_space.n

        # Push sparse_reward down into MarioSparseRewardWrapper if present.
        # (Outer wrappers don't forward extra positional args, so we set it as
        # an attribute on the wrapper instead of passing it per-step.)
        from SASR.utils import get_sparse_reward_wrapper
        _srw = get_sparse_reward_wrapper(self.env)
        if _srw is not None:
            _srw.sparse_reward = self.sparse_reward

        # * for the SAC-Discrete backbone
        self.actor = actor_class(self.env).to(self.device)
        self.qf_1 = critic_class(self.env).to(self.device)
        self.qf_2 = critic_class(self.env).to(self.device)
        self.qf_1_target = critic_class(self.env).to(self.device)
        self.qf_2_target = critic_class(self.env).to(self.device)

        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr, eps=eps)
        self.q_optimizer = optim.Adam(list(self.qf_1.parameters()) + list(self.qf_2.parameters()), lr=q_lr, eps=eps)

        # Entropy tuning for discrete actions
        self.alpha_autotune = alpha_autotune
        if alpha_autotune:
            # target entropy = -log(1/|A|) * 0.98 (standard for SAC-Discrete)
            self.target_entropy = -np.log(1.0 / self.num_actions) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        # initialize the replay buffer
        self.env.observation_space.dtype = np.float32
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False,
        )

        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # * tensorboard writer
        run_name = "{}-{}-{}-{}".format(
            exp_name,
            env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') and env.unwrapped.spec is not None else "mario",
            seed,
            datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S"),
        )
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        # * for the SASR mechanism
        self.reward_weight = reward_weight / 2
        self.feature_dim = feature_dim

        # S/F buffers store raw observations (not pre-extracted features).
        # Feature tensors are rebuilt every feature_refresh_interval optimize steps
        # so that buffer features and query features always use the same network weights,
        # fixing the staleness / feature-space mismatch problem.
        self.S_buffer = []
        self.S_buffer_tensor = torch.zeros(0, feature_dim).to(self.device)
        self.F_buffer = []
        self.F_buffer_tensor = torch.zeros(0, feature_dim).to(self.device)
        self.retention_rate = retention_rate
        self.feature_refresh_interval = feature_refresh_interval

        self.kde_bandwidth = kde_bandwidth
        self.kde_sample_burn_in = kde_sample_burn_in
        self.obs_dim = feature_dim  # KDE operates on feature space

        # RFF mapping for KDE acceleration
        if rff_dim is None:
            self.rff = False
        else:
            self.rff = True
            self.rff_dim = rff_dim
            self.rff_W = torch.randn(rff_dim, self.obs_dim).to(self.device) / kde_bandwidth
            self.rff_b = torch.rand(rff_dim).to(self.device) * 2 * torch.pi

    def _extract_features(self, obs_tensor):
        """Extract CNN features from image observations for KDE.
        Uses the actor's feature extractor (detached, no gradients).
        :param obs_tensor: (batch, C, H, W) float32 tensor
        :return: (batch, feature_dim) tensor
        """
        with torch.no_grad():
            features = self.actor.get_features(obs_tensor)
        return features

    def _extract_features_in_batches(self, obs_list, batch_size=2048):
        """Extract CNN features from a list of raw observations using the current network.
        Processes in mini-batches to avoid OOM.
        :param obs_list: list of numpy arrays, each (C, H, W)
        :param batch_size: processing batch size
        :return: (N, feature_dim) tensor
        """
        all_features = []
        for i in range(0, len(obs_list), batch_size):
            batch = np.array(obs_list[i:i + batch_size])
            batch_tensor = torch.FloatTensor(batch).to(self.device)
            all_features.append(self._extract_features(batch_tensor))
        return torch.cat(all_features, dim=0)

    def update_S(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]

        # Store raw observations; features are re-extracted with the current network
        # in _rebuild_buffer_features() to avoid stale feature-space mismatch.
        self.S_buffer += trajectory

    def update_F(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]

        # Store raw observations; features are re-extracted with the current network
        # in _rebuild_buffer_features() to avoid stale feature-space mismatch.
        self.F_buffer += trajectory

    def _rebuild_buffer_features(self, batch_size=2048):
        """Re-extract CNN features for all stored raw observations using the current network.
        Calling this periodically ensures buffer tensors and KDE query features
        are always in the same (up-to-date) feature space.
        :param batch_size: processing batch size for feature extraction
        """
        if len(self.S_buffer) > 0:
            self.S_buffer_tensor = self._extract_features_in_batches(self.S_buffer, batch_size)
        if len(self.F_buffer) > 0:
            self.F_buffer_tensor = self._extract_features_in_batches(self.F_buffer, batch_size)

    def KDE_RFF_sample(self, buffer, batch):
        if buffer.shape[0] <= self.kde_sample_burn_in:
            return torch.zeros(batch.shape[0]).to(self.device)

        z_buffer = math.sqrt(2 / self.rff_dim) * torch.cos(torch.matmul(buffer, self.rff_W.T) + self.rff_b)
        z_batch = math.sqrt(2 / self.rff_dim) * torch.cos(torch.matmul(batch, self.rff_W.T) + self.rff_b)

        kde_estimates = torch.sum(torch.matmul(z_buffer, z_batch.T), dim=0)

        return kde_estimates

    def KDE_sample(self, buffer, batch):
        if buffer.shape[0] <= self.kde_sample_burn_in:
            return torch.zeros(batch.shape[0]).to(self.device)

        distances_squared = torch.sum((batch[:, None, :] - buffer[None, :, :]) ** 2, dim=2)

        kernel_values = (1 / (2 * torch.pi * self.kde_bandwidth ** 2) ** (self.obs_dim / 2)) * torch.exp(
            -distances_squared / (2 * self.kde_bandwidth ** 2))

        kde_estimates = torch.sum(kernel_values, dim=1)

        return kde_estimates

    def learn(self, total_episodes=2000, learning_starts=10000, print_frequency=0):
        """Train for `total_episodes` episodes. `learning_starts` is still measured
        in transitions (timesteps) since it controls replay-buffer burn-in."""

        obs, _ = self.env.reset()

        # trajectory tracking (stores raw observations for CNN feature extraction later)
        trajectory = []
        flag_get = False
        episode_count = 0
        episode_returns = []
        global_step = 0

        # _current_epoch is the x-axis used by self.optimize(...) for TB logging.
        self._current_epoch = 0

        pbar = tqdm(total=total_episodes, desc="SASR-Discrete Learning (episodes)")
        while episode_count < total_episodes:
            if global_step < learning_starts:
                action = self.env.action_space.sample()
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, _, _ = self.actor.get_action(obs_tensor)
                action = action.item()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if "episode" in info:
                episode_count += 1
                self._current_epoch = episode_count
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], episode_count)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], episode_count)
                episode_returns.append(info["episode"]["r"])
                pbar.update(1)
                if print_frequency > 0 and episode_count % print_frequency == 0:
                    avg_return = np.mean(episode_returns[-print_frequency:])
                    print(f"Episode {episode_count} | Step {global_step} | "
                          f"Avg Return (last {print_frequency}): {avg_return:.2f} | "
                          f"Last Return: {info['episode']['r']:.2f}")

            self.replay_buffer.add(obs, next_obs, np.array([action]), reward, done, info)
            trajectory.append(obs)

            # Track success: flag_get means level completed
            if info.get("flag_get", False):
                flag_get = True

            if not done:
                obs = next_obs
            else:
                obs, _ = self.env.reset()

                # Classify trajectory as success or failure
                if flag_get:
                    self.update_S(trajectory)
                else:
                    self.update_F(trajectory)

                trajectory = []
                flag_get = False

            if global_step > learning_starts:
                self.optimize(global_step)

            global_step += 1

        pbar.close()
        self.env.close()
        self.writer.close()

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)

        # observations: (batch, C, H, W), actions: (batch, 1) integers
        observations = data.observations
        next_observations = data.next_observations
        actions = data.actions.long()  # (batch, 1)

        # Periodically rebuild S/F buffer feature tensors with the current network
        # so that buffer features and KDE query features share the same feature space.
        if global_step % self.feature_refresh_interval == 0:
            print("Rebuilding S/F buffer features at step {}...".format(global_step))
            self._rebuild_buffer_features()

        with torch.no_grad():
            # --- Next-state value computation (SAC-Discrete) ---
            # Actor outputs action probs for next state
            next_logits = self.actor(next_observations)
            next_action_probs = F.softmax(next_logits, dim=-1)
            next_log_action_probs = torch.log(next_action_probs + 1e-8)

            # Q-values from target networks for all actions
            qf_1_next_all = self.qf_1_target(next_observations)  # (batch, num_actions)
            qf_2_next_all = self.qf_2_target(next_observations)  # (batch, num_actions)
            min_qf_next = torch.min(qf_1_next_all, qf_2_next_all)

            # V(s') = sum_a [ pi(a|s') * (Q(s',a) - alpha * log pi(a|s')) ]
            next_v = (next_action_probs * (min_qf_next - self.alpha * next_log_action_probs)).sum(dim=-1)

            # --- SASR reward shaping ---
            # Extract features for KDE (use current observations)
            batch_features = self._extract_features(observations)

            density_values_S = self.KDE_RFF_sample(self.S_buffer_tensor,
                                                   batch_features) if self.rff else self.KDE_sample(
                self.S_buffer_tensor, batch_features)
            density_values_F = self.KDE_RFF_sample(self.F_buffer_tensor,
                                                   batch_features) if self.rff else self.KDE_sample(
                self.F_buffer_tensor, batch_features)
            pseudo_counts_S = density_values_S * global_step * self.retention_rate
            pseudo_counts_F = density_values_F * global_step * self.retention_rate
            alpha_param = torch.clamp(pseudo_counts_S + 1, min=1e-6)
            beta_param = torch.clamp(pseudo_counts_F + 1, min=1e-6)
            shaped_rewards = Beta(alpha_param, beta_param).sample()

            sasr_rewards = data.rewards.flatten() + self.reward_weight * shaped_rewards
            next_q_value = sasr_rewards + (1 - data.dones.flatten()) * self.gamma * next_v

        # --- Q-network loss ---
        # Q(s, a) = Q_all(s)[a]  (gather the Q-value for the taken action)
        qf_1_all = self.qf_1(observations)  # (batch, num_actions)
        qf_2_all = self.qf_2(observations)  # (batch, num_actions)
        qf_1_a_values = qf_1_all.gather(1, actions).squeeze(-1)  # (batch,)
        qf_2_a_values = qf_2_all.gather(1, actions).squeeze(-1)  # (batch,)

        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # --- Actor (policy) loss ---
        if global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                logits = self.actor(observations)
                action_probs = F.softmax(logits, dim=-1)
                log_action_probs = torch.log(action_probs + 1e-8)

                with torch.no_grad():
                    qf_1_pi = self.qf_1(observations)
                    qf_2_pi = self.qf_2(observations)
                    min_qf_pi = torch.min(qf_1_pi, qf_2_pi)

                # actor_loss = E_s[ sum_a pi(a|s) * (alpha * log pi(a|s) - Q(s,a)) ]
                actor_loss = (action_probs * (self.alpha * log_action_probs - min_qf_pi)).sum(dim=-1).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- Alpha (temperature) tuning ---
                if self.alpha_autotune:
                    # entropy = -sum_a pi(a|s) * log pi(a|s)
                    with torch.no_grad():
                        logits_alpha = self.actor(observations)
                        probs_alpha = F.softmax(logits_alpha, dim=-1)
                        log_probs_alpha = torch.log(probs_alpha + 1e-8)
                        entropy = -(probs_alpha * log_probs_alpha).sum(dim=-1)

                    alpha_loss = (self.log_alpha.exp() * (entropy - self.target_entropy)).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # --- Target network update ---
        if global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf_1.parameters(), self.qf_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf_2.parameters(), self.qf_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Logging ---
        # Sampling gate remains transition-based (fires every write_frequency
        # optimize() calls) so loss curves stay densely sampled, but the TB
        # x-axis is the current epoch so all scalars share an episode-count axis.
        if global_step % self.write_frequency == 0:
            epoch = getattr(self, "_current_epoch", 0)
            self.writer.add_scalar("losses/qf_1_values", qf_1_a_values.mean().item(), epoch)
            self.writer.add_scalar("losses/qf_2_values", qf_2_a_values.mean().item(), epoch)
            self.writer.add_scalar("losses/qf_1_loss", qf_1_loss.item(), epoch)
            self.writer.add_scalar("losses/qf_2_loss", qf_2_loss.item(), epoch)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, epoch)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), epoch)
            self.writer.add_scalar("losses/alpha", self.alpha, epoch)
            self.writer.add_scalar("sasr/s_buffer_size", len(self.S_buffer), epoch)
            self.writer.add_scalar("sasr/f_buffer_size", len(self.F_buffer), epoch)
            if self.alpha_autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), epoch)

    def save(self, indicator="best"):
        torch.save(self.actor.state_dict(),
                   os.path.join(self.save_folder, "actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        torch.save(self.qf_1.state_dict(),
                   os.path.join(self.save_folder, "qf_1-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        torch.save(self.qf_2.state_dict(),
                   os.path.join(self.save_folder, "qf_2-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))

    def evaluate(self, n_episodes=10):
        """Run deterministic policy evaluation, return flag_get success rate."""
        successes = 0
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            print(f"  [Eval] Episode {ep + 1}/{n_episodes}")
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.actor.get_deterministic_action(obs_tensor)
                obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                x_pos = info.get("x_pos", "N/A")
                print(f"    step={step:4d}  action={action.item()}  x_pos={x_pos}")
                step += 1
                if info.get("flag_get", False):
                    successes += 1
                    print(f"    >> Flag get!")
                    break
        return successes / n_episodes

    def curriculum_learn(self, learning_starts=10000, print_frequency=0,
                         min_stage_episodes=100, eval_interval=20,
                         eval_episodes=10, max_stage_episodes=500, pass_rate_threshold=0.5):
        """Curriculum learning: train from near-goal positions to far, stage by stage.

        The environment must be wrapped with CurriculumMarioWrapper.
        Network parameters, replay buffer, and S/F buffers are preserved across stages.

        Args:
            learning_starts: random exploration steps before policy is used
            print_frequency: print average return every N episodes (0 to disable)
            min_stage_episodes: minimum episodes per stage before evaluation starts
            eval_interval: evaluate every N episodes (after min_stage_episodes)
            eval_episodes: number of episodes per evaluation
            max_stage_episodes: force advance to next stage after this many episodes
            pass_rate_threshold: success rate threshold to advance to next stage
        """

        num_stages = self.env.num_stages
        global_step = 0
        total_episode_count = 0
        self._current_epoch = 0

        for stage_idx in range(num_stages):
            # Set environment to current curriculum stage
            self.env.set_stage(stage_idx)
            target_x = self.env.curriculum_positions[stage_idx][0]
            print("\n" + "=" * 60)
            print("  CURRICULUM STAGE {}/{}: start_x={}".format(stage_idx, num_stages - 1, target_x))
            print("=" * 60)

            obs, _ = self.env.reset()
            trajectory = []
            flag_get = False
            stage_episode_count = 0
            episode_returns = []
            passed = False

            while stage_episode_count < max_stage_episodes:
                # Select action
                if global_step < learning_starts:
                    action = self.env.action_space.sample()
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _, _ = self.actor.get_action(obs_tensor)
                    action = action.item()

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if "episode" in info:
                    stage_episode_count += 1
                    total_episode_count += 1
                    self._current_epoch = total_episode_count
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], total_episode_count)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], total_episode_count)
                    self.writer.add_scalar("curriculum/stage", stage_idx, total_episode_count)
                    episode_returns.append(info["episode"]["r"])
                    if print_frequency > 0 and stage_episode_count % print_frequency == 0:
                        avg_return = np.mean(episode_returns[-print_frequency:])
                        print("Stage {} | Episode {} | Step {} | "
                              "Avg Return (last {}): {:.2f} | "
                              "Last Return: {:.2f}".format(
                                  stage_idx, stage_episode_count, global_step,
                                  print_frequency, avg_return, info["episode"]["r"]))

                self.replay_buffer.add(obs, next_obs, np.array([action]), reward, done, info)
                trajectory.append(obs)

                if info.get("flag_get", False):
                    flag_get = True

                if not done:
                    obs = next_obs
                else:
                    obs, _ = self.env.reset()

                    # Classify trajectory as success or failure
                    if flag_get:
                        self.update_S(trajectory)
                    else:
                        self.update_F(trajectory)

                    trajectory = []
                    flag_get = False

                    # Evaluation check: after min episodes, every eval_interval episodes
                    # Skip evaluation if training hasn't started yet
                    if (global_step >= learning_starts and
                            stage_episode_count >= min_stage_episodes and
                            stage_episode_count % eval_interval == 0):
                        print("Start Eval")
                        pass_rate = self.evaluate(eval_episodes)
                        # Reset env after evaluation (evaluate leaves it in a done state)
                        obs, _ = self.env.reset()
                        trajectory = []
                        flag_get = False
                        self.writer.add_scalar("curriculum/pass_rate", pass_rate, total_episode_count)
                        print("  [Eval] Stage {} | Episode {} | Pass rate: {:.1f}%".format(
                            stage_idx, stage_episode_count, pass_rate * 100))
                        if pass_rate >= pass_rate_threshold:
                            print("  >>> Stage {} PASSED! (rate={:.1f}%)".format(
                                stage_idx, pass_rate * 100))
                            passed = True
                            break

                if global_step > learning_starts:
                    self.optimize(global_step)

                global_step += 1

            if not passed:
                print("  >>> Stage {} reached max episodes ({}), advancing.".format(
                    stage_idx, max_stage_episodes))

            # Save checkpoint after each stage
            self.save(indicator="stage{}".format(stage_idx))

        print("\nCurriculum training complete. Total steps: {}".format(global_step))
        self.env.close()
        self.writer.close()

    def subgoal_curriculum_learn(self, subgoal_thresholds, learning_starts=10000,
                                  print_frequency=0, min_stage_episodes=200,
                                  eval_window=50, max_stage_episodes=2000,
                                  success_rate_threshold=0.5):
        """Sub-goal curriculum learning: progressively increase the distance threshold for success.

        Mario always starts from position x=40. The agent must reach progressively
        farther distance thresholds (e.g. 20%, 40%, 60%, 80%, 100%) to advance stages.
        Success is evaluated by checking normalized_distance against the current threshold.

        Network parameters and replay buffer persist across stages.
        S/F buffers are cleared on stage transitions since the definition of
        "success" changes per stage.

        Args:
            subgoal_thresholds: list of float thresholds [0,1], e.g. [0.2, 0.4, 0.6, 0.8, 1.0]
            learning_starts: random exploration steps before policy is used
            print_frequency: print average return every N episodes (0 to disable)
            min_stage_episodes: minimum episodes per stage before stage advancement is evaluated
            eval_window: sliding window size for computing success rate
            max_stage_episodes: force advance to next stage after this many episodes
            success_rate_threshold: success rate threshold to advance to next stage
        """
        from SASR.utils import get_sparse_reward_wrapper

        # Find the MarioSparseRewardWrapper in the wrapper chain
        reward_wrapper = get_sparse_reward_wrapper(self.env)
        if reward_wrapper is None:
            raise ValueError("Could not find MarioSparseRewardWrapper in the environment wrapper chain. "
                             "Sub-goal curriculum requires MarioSparseRewardWrapper.")

        num_stages = len(subgoal_thresholds)
        global_step = 0

        for stage_idx in range(num_stages):
            threshold = subgoal_thresholds[stage_idx]

            # Clear S/F buffers: "success" semantics change per stage
            self._clear_sf_buffers()

            print("\n" + "=" * 60)
            print("  SUB-GOAL STAGE {}/{}: threshold={:.0f}% distance".format(
                stage_idx + 1, num_stages, threshold * 100))
            print("=" * 60)

            obs, _ = self.env.reset()
            trajectory = []
            stage_episode_count = 0
            episode_returns = []
            stage_successes = []  # sliding window of True/False for recent episodes
            passed = False

            while stage_episode_count < max_stage_episodes:

                # Select action
                if global_step < learning_starts:
                    action = self.env.action_space.sample()
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _, _ = self.actor.get_action(obs_tensor)
                    action = action.item()

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if "episode" in info:
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    self.writer.add_scalar("curriculum/stage", stage_idx, global_step)
                    self.writer.add_scalar("curriculum/subgoal_threshold", threshold, global_step)
                    stage_episode_count += 1
                    episode_returns.append(info["episode"]["r"])

                    # Determine success for this episode based on normalized_distance
                    normalized_dist = info.get("normalized_distance", 0.0)
                    ep_success = normalized_dist >= threshold
                    stage_successes.append(ep_success)

                    if print_frequency > 0 and stage_episode_count % print_frequency == 0:
                        avg_return = np.mean(episode_returns[-print_frequency:])
                        recent_rate = (np.mean(stage_successes[-eval_window:]) * 100
                                       if stage_successes else 0.0)
                        print("Stage {}/{} | Episode {} | Step {} | "
                              "Avg Return (last {}): {:.3f} | "
                              "Last dist: {:.1f}% | "
                              "Success rate (last {}): {:.1f}%".format(
                                  stage_idx + 1, num_stages,
                                  stage_episode_count, global_step,
                                  print_frequency, avg_return,
                                  normalized_dist * 100,
                                  eval_window, recent_rate))

                self.replay_buffer.add(obs, next_obs, np.array([action]), reward, done, info)
                trajectory.append(obs)

                if not done:
                    obs = next_obs
                else:
                    obs, _ = self.env.reset()

                    # Classify trajectory: success if normalized_distance >= threshold
                    normalized_dist = info.get("normalized_distance", 0.0)
                    if normalized_dist >= threshold:
                        self.update_S(trajectory)
                    else:
                        self.update_F(trajectory)

                    trajectory = []

                    # Stage advancement check (sliding window)
                    if (global_step >= learning_starts and
                            stage_episode_count >= min_stage_episodes and
                            len(stage_successes) >= eval_window):
                        recent_success_rate = np.mean(stage_successes[-eval_window:])
                        self.writer.add_scalar("curriculum/stage_success_rate",
                                               recent_success_rate, global_step)

                        if recent_success_rate >= success_rate_threshold:
                            print("\n  >>> Stage {}/{} PASSED! "
                                  "(success rate={:.1f}% over last {} episodes)".format(
                                      stage_idx + 1, num_stages,
                                      recent_success_rate * 100, eval_window))
                            passed = True
                            break

                if global_step > learning_starts:
                    self.optimize(global_step)

                global_step += 1

            if not passed:
                print("  >>> Stage {}/{} reached max episodes ({}), advancing.".format(
                    stage_idx + 1, num_stages, max_stage_episodes))

            # Save checkpoint after each stage
            self.save(indicator="subgoal_stage{}".format(stage_idx))

        print("\nSub-goal curriculum training complete. Total steps: {}".format(global_step))
        self.env.close()
        self.writer.close()

    def _clear_sf_buffers(self):
        """Clear the success/failure buffers and their feature tensors."""
        self.S_buffer = []
        self.S_buffer_tensor = torch.zeros(0, self.feature_dim).to(self.device)
        self.F_buffer = []
        self.F_buffer_tensor = torch.zeros(0, self.feature_dim).to(self.device)