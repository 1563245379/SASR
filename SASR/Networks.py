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
