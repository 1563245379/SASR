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
                 feature_dim=512):
        """
        Initialize the SASR algorithm for discrete action spaces.
        :param env: the gymnasium-compatible environment with image observations and discrete actions
        :param actor_class: the actor class (SACActorDiscrete)
        :param critic_class: the critic class (QNetworkDiscrete)
        :param feature_dim: dimension of CNN feature vectors used for KDE (default: 512)
        """

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

        # S/F buffers store CNN feature vectors (not raw images)
        self.S_buffer = []
        self.S_buffer_tensor = torch.Tensor(self.S_buffer).to(self.device)
        self.F_buffer = []
        self.F_buffer_tensor = torch.Tensor(self.F_buffer).to(self.device)
        self.retention_rate = retention_rate

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

    def _extract_features_from_numpy(self, obs_list):
        """Extract features from a list of numpy observations.
        Processes in batches to avoid OOM.
        :param obs_list: list of numpy arrays, each (C, H, W)
        :return: list of numpy arrays, each (feature_dim,)
        """
        if len(obs_list) == 0:
            return []
        batch = np.array(obs_list)
        batch_tensor = torch.FloatTensor(batch).to(self.device)
        features = self._extract_features(batch_tensor)
        return features.cpu().numpy().tolist()

    def update_S(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]

        # Extract CNN features and store them
        feature_list = self._extract_features_from_numpy(trajectory)
        self.S_buffer += feature_list
        self.S_buffer_tensor = torch.Tensor(np.array(self.S_buffer)).to(self.device)

    def update_F(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]

        feature_list = self._extract_features_from_numpy(trajectory)
        self.F_buffer += feature_list
        self.F_buffer_tensor = torch.Tensor(np.array(self.F_buffer)).to(self.device)

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

    def learn(self, total_timesteps=1000000, learning_starts=10000, print_frequency=0):

        obs, _ = self.env.reset()

        # trajectory tracking (stores raw observations for CNN feature extraction later)
        trajectory = []
        flag_get = False
        episode_count = 0
        episode_returns = []

        for global_step in tqdm(range(total_timesteps), desc="SASR-Discrete Learning"):
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
                episode_count += 1
                episode_returns.append(info["episode"]["r"])
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

        self.env.close()
        self.writer.close()

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)

        # observations: (batch, C, H, W), actions: (batch, 1) integers
        observations = data.observations
        next_observations = data.next_observations
        actions = data.actions.long()  # (batch, 1)

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
        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/qf_1_values", qf_1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_2_values", qf_2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_1_loss", qf_1_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_2_loss", qf_2_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/alpha", self.alpha, global_step)
            self.writer.add_scalar("sasr/s_buffer_size", len(self.S_buffer), global_step)
            self.writer.add_scalar("sasr/f_buffer_size", len(self.F_buffer), global_step)
            if self.alpha_autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    def save(self, indicator="best"):
        torch.save(self.actor.state_dict(),
                   os.path.join(self.save_folder, "actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        torch.save(self.qf_1.state_dict(),
                   os.path.join(self.save_folder, "qf_1-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        torch.save(self.qf_2.state_dict(),
                   os.path.join(self.save_folder, "qf_2-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
