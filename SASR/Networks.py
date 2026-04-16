"""
The networks used in the SASR algorithm for the continuous control tasks.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class QNetworkContinuousControl(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOActorCriticContinuous(nn.Module):
    """
    Shared trunk actor–critic for continuous Box actions.
    Gaussian policy in pre-tanh space with tanh squashing to env bounds (same construction as SACActor).
    """

    def __init__(self, env, hidden_dim=256, num_hidden_layers=2, actor_log_std_init=0.0):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        self.hidden = nn.ModuleList()
        d = obs_dim
        for _ in range(num_hidden_layers):
            self.hidden.append(nn.Linear(d, hidden_dim))
            d = hidden_dim

        self.actor_mean = nn.Linear(d, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((1, act_dim), float(actor_log_std_init)))
        self.critic = nn.Linear(d, 1)
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.log_std_max = 2.0
        self.log_std_min = -5.0

    def _base(self, x):
        for lin in self.hidden:
            x = F.relu(lin(x))
        return x

    def get_action_and_value(self, obs, action=None):
        """
        If action is None: sample an action, return logprob, entropy, value.
        If action is provided: compute logprob and entropy for that action (for PPO update).
        obs: (batch, obs_dim)
        action: (batch, act_dim) in env space, or None
        """
        x = self._base(obs)
        mean = self.actor_mean(x)
        raw_log_std = self.actor_log_std.expand_as(mean)
        log_std = torch.tanh(raw_log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        if action is None:
            x_t = normal.rsample()
        else:
            y_t = (action - self.action_bias) / (self.action_scale + 1e-8)
            y_t = torch.clamp(y_t, -0.999999, 0.999999)
            x_t = torch.atanh(y_t)

        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1)
        entropy = normal.entropy().sum(-1)

        value = self.critic(x)
        action_out = y_t * self.action_scale + self.action_bias if action is None else action
        return action_out, log_prob, entropy, value.squeeze(-1)

    def get_deterministic_action(self, obs):
        """Mean policy (tanh of actor mean, then scale), same convention as SACActor.get_action."""
        x = self._base(obs)
        mean = self.actor_mean(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean


# ============================================================
# CNN-based networks for discrete action spaces (e.g., Mario)
# ============================================================

class CNNFeatureExtractor(nn.Module):
    """Shared CNN backbone for image observations.
    Input: (batch, 4, 84, 84) float32 [0,1]
    Output: (batch, 512) feature vector
    """

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(3136, 512)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x


class SACActorDiscrete(nn.Module):
    """SAC-Discrete actor: outputs action probabilities over discrete actions.
    Input: image observation (batch, C, H, W)
    Output via get_action: (action, log_prob, action_probs)
    """

    def __init__(self, env):
        super().__init__()
        in_channels = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.feature_extractor = CNNFeatureExtractor(in_channels)
        self.fc_logits = nn.Linear(512, self.num_actions)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.fc_logits(features)
        return logits

    def get_features(self, x):
        return self.feature_extractor(x)

    def get_action(self, x):
        logits = self.forward(x)
        # Compute action probs with numerical stability
        action_probs = F.softmax(logits, dim=-1)
        # Avoid log(0)
        log_action_probs = torch.log(action_probs + 1e-8)

        # Sample action from categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        # Per-action log_prob (for the sampled action)
        log_prob = log_action_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob, action_probs

    def get_deterministic_action(self, x):
        logits = self.forward(x)
        action_probs = F.softmax(logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)
        return action


class QNetworkDiscrete(nn.Module):
    """Q-network for discrete actions: outputs Q-value for each action.
    Input: image observation (batch, C, H, W)
    Output: (batch, num_actions)
    """

    def __init__(self, env):
        super().__init__()
        in_channels = env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.feature_extractor = CNNFeatureExtractor(in_channels)
        self.fc_q = nn.Linear(512, num_actions)

    def forward(self, x):
        features = self.feature_extractor(x)
        q_values = self.fc_q(features)
        return q_values


class PPOActorCriticDiscrete(nn.Module):
    """
    Shared CNN trunk + discrete policy (Categorical) + scalar value for PPO on image observations.
    Input: (batch, C, H, W) float32; actions are integers in [0, num_actions).
    """

    def __init__(self, env):
        super().__init__()
        in_channels = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.features = CNNFeatureExtractor(in_channels)
        self.actor_head = nn.Linear(512, self.num_actions)
        self.critic_head = nn.Linear(512, 1)

    def get_action_and_value(self, obs, action=None):
        """
        obs: (batch, C, H, W)
        action: optional (batch,) long — indices of discrete actions taken during rollout.
        """
        h = self.features(obs)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()
        else:
            action = action.long().view(-1)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs):
        h = self.features(obs)
        logits = self.actor_head(h)
        return logits.argmax(dim=-1)
