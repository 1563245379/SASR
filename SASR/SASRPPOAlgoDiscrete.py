"""
PPO + SASR shaped reward for discrete-action / image-observation environments (e.g. Super Mario Bros).

PPO backbone is identical to PPOAlgoDiscrete.PPODiscrete (CleanRL-style: GAE-Lambda, clipped
surrogate, single-env rollout). SASR shaping is identical to SASRAlgoDiscrete: KDE pseudo-counts
over per-trajectory CNN features split into Success (S) / Failure (F) buffers (success = the
episode's info["flag_get"] was ever True), with a Beta(alpha, beta).sample() shaped reward.

The shaped reward is added into the per-step env reward BEFORE GAE so it propagates into both
advantages and value targets, matching SASR-Discrete's TD-target semantics
(SASRAlgoDiscrete.py:302).

Deliberate divergences from the reference SASR-Discrete (each documented inline):
- KDE features come from the PPO actor-critic's shared CNN (`self.ac.features`) instead of a
  separate SAC actor's `get_features`. The CNN is trained by PPO's policy/value loss; SASR uses
  it under no_grad.
- S/F buffers are FIFO-capped at `buffer_max_size` to bound memory and keep the per-update
  re-tensorize cost from going quadratic over a 3M-step Mario run.
"""

import os
import datetime
import time
import math
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.beta import Beta

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


class SASRPPODiscrete:
    def __init__(
        self,
        env,
        actor_critic_class,
        exp_name="sasr-ppo-mario",
        seed=1,
        cuda=0,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        eps=1e-5,
        steps_per_rollout=2048,
        num_minibatches=32,
        update_epochs=10,
        anneal_lr=True,
        write_frequency=100,
        save_folder="./sasr-ppo-mario/",
        # SASR-specific
        reward_weight=0.6,
        kde_bandwidth=0.2,
        kde_sample_burn_in=1000,
        rff_dim=1000,
        retention_rate=0.1,
        feature_dim=512,
        buffer_max_size=50000,
        feature_chunk_size=256,
        shaping_warmup_steps=0,
        success_criterion="flag_or_quantile",
        success_quantile=0.75,
        success_window=100,
        success_min_samples=5,
        feature_normalize=True,
        s_buffer_max_size=None,
        f_buffer_max_size=None,
    ):
        self.env = env
        self.exp_name = exp_name
        self.seed = seed
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.steps_per_rollout = steps_per_rollout
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.anneal_lr = anneal_lr
        self.write_frequency = write_frequency
        self.save_folder = save_folder

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.ac = actor_critic_class(env).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate, eps=eps)
        self.initial_lr = learning_rate

        self.env.observation_space.dtype = np.float32

        _spec = getattr(env, "spec", None)
        env_tag = _spec.id if _spec is not None else "MarioDiscrete"
        run_name = "{}-{}-{}-{}".format(
            exp_name,
            env_tag,
            seed,
            datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S"),
        )
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))

        os.makedirs(self.save_folder, exist_ok=True)

        # SASR state — reward_weight is halved here (matches SASRAlgoDiscrete.py:122) and used
        # once at injection time. Do not multiply by 0.5 again at the splice point.
        self.reward_weight = reward_weight / 2
        self.feature_dim = feature_dim
        self.buffer_max_size = buffer_max_size
        self.feature_chunk_size = feature_chunk_size
        self.shaping_warmup_steps = shaping_warmup_steps

        # Success-classification: deciding whether a finished trajectory feeds S or F. The
        # reference SASR-Discrete uses info["flag_get"] only, but on hard Mario levels the
        # agent may never reach the flag during training, leaving S permanently empty and
        # the shaping gate permanently closed. The default here is flag_or_quantile, which
        # adds an adaptive elite-relative split based on a rolling window of terminal
        # rewards (MarioSparseRewardWrapper emits normalized distance ∈ [0, 1] at terminal).
        if success_criterion not in ("flag", "quantile", "flag_or_quantile"):
            raise ValueError(
                "success_criterion must be one of: flag | quantile | flag_or_quantile"
            )
        if not (0.0 < success_quantile < 1.0):
            raise ValueError("success_quantile must be in (0, 1)")
        self.success_criterion = success_criterion
        self.success_quantile = success_quantile
        self.success_min_samples = success_min_samples
        self._terminal_rewards = collections.deque(maxlen=success_window)
        self._last_threshold = float("nan")
        self._last_density_gap = float("nan")
        self._last_beta_concentration = float("nan")

        # KDE feature normalization: CNN features have un-bounded scale that drifts as the
        # trunk learns, which makes a fixed kde_bandwidth meaningless. L2-normalizing each
        # feature vector turns the RBF / RFF kernel into a cosine-distance kernel — the
        # bandwidth then carries a stable interpretation (≈ angular-similarity threshold)
        # across all of training. With normalization, kde_bandwidth=0.2 corresponds to
        # ||x-y||² ≈ 2(1-cos), so cos > 1-0.04 ≈ 0.96 ⇒ "very similar features". Without
        # normalization, the same bandwidth makes RFF density estimates degenerate to noise
        # in 512-dim space, which is the dominant reason SASR shaping was inert in early
        # diagnostics (terminal-reward spread tiny, S/F buffers indistinguishable in KDE).
        self.feature_normalize = feature_normalize

        self.S_buffer = []
        self.S_buffer_tensor = torch.empty(0, feature_dim, device=self.device)
        self.F_buffer = []
        self.F_buffer_tensor = torch.empty(0, feature_dim, device=self.device)
        self.retention_rate = retention_rate
        # Per-buffer FIFO caps. Defaults to buffer_max_size for backward compatibility, but
        # smaller S keeps the success buffer "fresh" and prevents stale-elite lock-in: as
        # the policy evolves, old entries computed under earlier CNN weights get evicted
        # before they can pull the policy backward.
        self.s_buffer_max_size = s_buffer_max_size if s_buffer_max_size is not None else buffer_max_size
        self.f_buffer_max_size = f_buffer_max_size if f_buffer_max_size is not None else buffer_max_size

        self.kde_bandwidth = kde_bandwidth
        self.kde_sample_burn_in = kde_sample_burn_in
        self.obs_dim = feature_dim

        if rff_dim is None:
            self.rff = False
            self.rff_dim = None
            self.rff_W = None
            self.rff_b = None
        else:
            self.rff = True
            self.rff_dim = rff_dim
            self.rff_W = torch.randn(rff_dim, self.obs_dim, device=self.device) / kde_bandwidth
            self.rff_b = torch.rand(rff_dim, device=self.device) * 2 * torch.pi

        # Trajectory state lives on the instance so it carries across rollout boundaries.
        self._cur_traj = []
        self._cur_flag_get = False

    def _extract_features(self, obs_tensor):
        """Run the PPO actor-critic's shared CNN with no gradients. obs_tensor: (B, C, H, W) float32.

        When feature_normalize is True, L2-normalize each feature vector so the KDE bandwidth
        carries a consistent cosine-distance interpretation regardless of CNN-output scale.
        """
        with torch.no_grad():
            feats = self.ac.features(obs_tensor)
            if self.feature_normalize:
                feats = F.normalize(feats, p=2, dim=-1, eps=1e-8)
            return feats

    def _extract_features_from_numpy(self, obs_list):
        """Extract features from a list of (C, H, W) numpy arrays. Chunked to bound VRAM."""
        if len(obs_list) == 0:
            return []
        out = []
        chunk = max(1, self.feature_chunk_size)
        for i in range(0, len(obs_list), chunk):
            batch = np.asarray(obs_list[i:i + chunk], dtype=np.float32)
            batch_t = torch.from_numpy(batch).to(self.device)
            feats = self._extract_features(batch_t).cpu().numpy()
            out.extend(list(feats))
        return out

    def _trim_buffer(self, buffer_list, max_size):
        """FIFO-cap a buffer list in place. Drops the oldest entries when over capacity."""
        if max_size is None or max_size <= 0:
            return buffer_list
        if len(buffer_list) > max_size:
            del buffer_list[: len(buffer_list) - max_size]
        return buffer_list

    def _classify_trajectory(self, terminal_reward, flag_get):
        """Decide whether the just-finished trajectory should feed S (success) or F (failure).

        Modes:
        - 'flag'              : S iff flag_get (reference SASR-Discrete behavior).
        - 'quantile'          : S iff terminal_reward is in the top (1 - success_quantile)
                                of the rolling window. Adaptive — never starves S of data.
        - 'flag_or_quantile'  : flag_get OR quantile pass.

        Until the rolling window has at least success_min_samples entries the quantile
        rule is disabled (returns False) so we don't classify off a single noisy sample.
        """
        if self.success_criterion == "flag":
            return bool(flag_get)

        quantile_success = False
        if len(self._terminal_rewards) >= self.success_min_samples:
            threshold = float(np.quantile(np.array(self._terminal_rewards, dtype=np.float64),
                                          self.success_quantile))
            self._last_threshold = threshold
            quantile_success = terminal_reward >= threshold

        if self.success_criterion == "quantile":
            return quantile_success
        # flag_or_quantile
        return bool(flag_get) or quantile_success

    def update_S(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]
        feature_list = self._extract_features_from_numpy(trajectory)
        self.S_buffer += feature_list
        self._trim_buffer(self.S_buffer, self.s_buffer_max_size)
        self.S_buffer_tensor = torch.as_tensor(
            np.asarray(self.S_buffer, dtype=np.float32), device=self.device
        )

    def update_F(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]
        feature_list = self._extract_features_from_numpy(trajectory)
        self.F_buffer += feature_list
        self._trim_buffer(self.F_buffer, self.f_buffer_max_size)
        self.F_buffer_tensor = torch.as_tensor(
            np.asarray(self.F_buffer, dtype=np.float32), device=self.device
        )

    def KDE_RFF_sample(self, buffer, batch):
        if buffer.shape[0] <= self.kde_sample_burn_in:
            return torch.zeros(batch.shape[0], device=self.device)

        z_buffer = math.sqrt(2 / self.rff_dim) * torch.cos(torch.matmul(buffer, self.rff_W.T) + self.rff_b)
        z_batch = math.sqrt(2 / self.rff_dim) * torch.cos(torch.matmul(batch, self.rff_W.T) + self.rff_b)

        kde_estimates = torch.sum(torch.matmul(z_buffer, z_batch.T), dim=0)
        return kde_estimates

    def KDE_sample(self, buffer, batch):
        if buffer.shape[0] <= self.kde_sample_burn_in:
            return torch.zeros(batch.shape[0], device=self.device)

        distances_squared = torch.sum((batch[:, None, :] - buffer[None, :, :]) ** 2, dim=2)
        kernel_values = (1 / (2 * torch.pi * self.kde_bandwidth ** 2) ** (self.obs_dim / 2)) * torch.exp(
            -distances_squared / (2 * self.kde_bandwidth ** 2)
        )
        kde_estimates = torch.sum(kernel_values, dim=1)
        return kde_estimates

    def _compute_shaped_rewards(self, mb_obs_np, global_step):
        """Compute SASR shaped reward for an entire rollout buffer.

        mb_obs_np: (T, C, H, W) float32. Returns (shaped, active) where shaped is a (T,)
        float32 numpy array to add into mb_rewards before GAE, and active is a bool flag
        indicating whether shaping engaged this rollout.

        Mitigation against action collapse: until S buffer crosses kde_sample_burn_in
        (i.e. before the agent has produced enough successful trajectories), KDE_*_sample
        returns zeros for both densities. That collapses Beta(alpha, beta) to Beta(1, 1) =
        Uniform(0, 1), which is per-step uniform noise — far stronger than the sparse env
        reward and a reliable driver of premature policy entropy collapse. Skip shaping
        entirely until S has real data, and optionally also wait `shaping_warmup_steps`
        env steps for an explicit warm start.
        """
        T = mb_obs_np.shape[0]

        if self.S_buffer_tensor.shape[0] <= self.kde_sample_burn_in:
            return np.zeros(T, dtype=np.float32), False
        if global_step < self.shaping_warmup_steps:
            return np.zeros(T, dtype=np.float32), False

        chunk = max(1, self.feature_chunk_size)
        feats_chunks = []
        for i in range(0, T, chunk):
            batch_t = torch.from_numpy(mb_obs_np[i:i + chunk]).to(self.device)
            feats_chunks.append(self._extract_features(batch_t))
        batch_features = torch.cat(feats_chunks, dim=0)  # (T, feature_dim)

        if self.rff:
            density_S = self.KDE_RFF_sample(self.S_buffer_tensor, batch_features)
            density_F = self.KDE_RFF_sample(self.F_buffer_tensor, batch_features)
        else:
            density_S = self.KDE_sample(self.S_buffer_tensor, batch_features)
            density_F = self.KDE_sample(self.F_buffer_tensor, batch_features)

        # Pseudo-count multiplier: the reference SASR-Discrete uses `global_step * retention_rate`,
        # which is correct only under the implicit assumption that buffers grow without bound
        # (i.e. len(buffer) ≈ global_step * retention_rate). With the FIFO cap introduced here,
        # that invariant breaks once buffers are full: the multiplier keeps growing while
        # len(buffer) plateaus, so alpha+beta is artificially inflated, Beta becomes overly
        # concentrated, and the deterministic ratio density_S/(density_S+density_F) starts
        # dominating — amplifying any stale-elite bias in the S buffer into a hard policy
        # constraint. Replacing with len(buffer) restores the reference invariant under caps:
        # alpha+beta ≈ 1 + (#comparable_S) + (#comparable_F), bounded by buffer_max_size.
        s_count = float(len(self.S_buffer))
        f_count = float(len(self.F_buffer))
        pseudo_counts_S = density_S * s_count * self.retention_rate
        pseudo_counts_F = density_F * f_count * self.retention_rate
        alpha_param = torch.clamp(pseudo_counts_S + 1, min=1e-6)
        beta_param = torch.clamp(pseudo_counts_F + 1, min=1e-6)
        shaped = Beta(alpha_param, beta_param).sample()

        # Diagnostic: average per-step (density_S - density_F). Near zero ⇒ S and F buffers
        # are KDE-indistinguishable and shaping is just Beta noise around 0.5. Positive ⇒ SASR
        # is rewarding success-typical states (intended). Negative ⇒ SASR is pushing the
        # policy AWAY from S manifold (regression — usually means S is stale).
        with torch.no_grad():
            self._last_density_gap = float((density_S - density_F).mean().item())
            self._last_beta_concentration = float((alpha_param + beta_param).mean().item())

        return shaped.detach().cpu().numpy().astype(np.float32), True

    def learn(self, total_timesteps=1_000_000):
        obs_shape = self.env.observation_space.shape

        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        global_step = 0
        num_updates = max(1, int(math.ceil(total_timesteps / self.steps_per_rollout)))
        pbar = tqdm(range(num_updates), desc="SASR-PPO-Mario Learning")

        last_shaped_mean = 0.0
        last_shaping_active = 0.0

        for update in pbar:
            if self.anneal_lr:
                frac = 1.0 - (update / max(1, num_updates - 1))
                self.optimizer.param_groups[0]["lr"] = frac * self.initial_lr

            mb_obs = np.zeros((self.steps_per_rollout, *obs_shape), dtype=np.float32)
            mb_actions = np.zeros((self.steps_per_rollout,), dtype=np.int64)
            mb_logprobs = np.zeros((self.steps_per_rollout,), dtype=np.float32)
            mb_rewards = np.zeros((self.steps_per_rollout,), dtype=np.float32)
            mb_dones = np.zeros((self.steps_per_rollout,), dtype=np.float32)
            mb_values = np.zeros((self.steps_per_rollout,), dtype=np.float32)

            for step in range(self.steps_per_rollout):
                global_step += 1
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    action, logprob, _, value = self.ac.get_action_and_value(obs_t)
                    action = action.detach()
                    logprob = logprob.detach()
                    value = value.detach()

                mb_obs[step] = obs
                mb_actions[step] = int(action.cpu().numpy().reshape(-1)[0])
                mb_logprobs[step] = float(logprob.cpu().numpy().reshape(-1)[0])
                mb_values[step] = float(value.cpu().numpy().reshape(-1)[0])

                # SASR: append the obs we just acted on to the in-flight trajectory.
                self._cur_traj.append(obs.copy())

                next_obs, reward, terminated, truncated, info = self.env.step(int(mb_actions[step]))
                done = float(terminated or truncated)

                if info.get("flag_get", False):
                    self._cur_flag_get = True

                if "episode" in info:
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                mb_rewards[step] = reward
                mb_dones[step] = done

                obs = np.asarray(next_obs, dtype=np.float32)
                if done:
                    obs, _ = self.env.reset()
                    obs = np.asarray(obs, dtype=np.float32)

                    # Flush completed trajectory into S or F. Terminal reward is the
                    # current step's reward (MarioSparseRewardWrapper emits the only
                    # non-zero value at done/truncated).
                    terminal_reward = float(reward)
                    self._terminal_rewards.append(terminal_reward)
                    is_success = self._classify_trajectory(terminal_reward, self._cur_flag_get)
                    if is_success:
                        self.update_S(self._cur_traj)
                    else:
                        self.update_F(self._cur_traj)
                    self._cur_traj = []
                    self._cur_flag_get = False

            # SASR shaping: compute per-step shaped rewards over the whole rollout, then add
            # into mb_rewards before GAE. Shaping is gated until S buffer is non-degenerate
            # (see _compute_shaped_rewards); early rollouts pass through unchanged.
            shaped, shaping_active = self._compute_shaped_rewards(mb_obs, global_step)
            mb_rewards = mb_rewards + self.reward_weight * shaped
            last_shaped_mean = float(shaped.mean())
            last_shaping_active = 1.0 if shaping_active else 0.0

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_v = self.ac.get_action_and_value(obs_t)[3]
            next_v = float(next_v.cpu().numpy().reshape(-1)[0])

            advantages = np.zeros((self.steps_per_rollout,), dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(self.steps_per_rollout)):
                if t == self.steps_per_rollout - 1:
                    v_next = next_v
                else:
                    v_next = mb_values[t + 1]
                nextvalues = (1.0 - mb_dones[t]) * v_next
                delta = mb_rewards[t] + self.gamma * nextvalues - mb_values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * (1.0 - mb_dones[t]) * lastgaelam
                advantages[t] = lastgaelam

            returns = advantages + mb_values

            b_obs = torch.as_tensor(mb_obs, device=self.device)
            b_actions = torch.as_tensor(mb_actions, dtype=torch.long, device=self.device)
            b_logprobs = torch.as_tensor(mb_logprobs, device=self.device)
            b_advantages = torch.as_tensor(advantages, device=self.device)
            b_returns = torch.as_tensor(returns, device=self.device)

            adv_std = b_advantages.std()
            b_advantages = (b_advantages - b_advantages.mean()) / (adv_std + 1e-8)

            batch_size = self.steps_per_rollout
            minibatch_size = max(1, batch_size // self.num_minibatches)

            inds = np.arange(batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]

                    _, newlogprob, entropy, newvalue = self.ac.get_action_and_value(b_obs[idx], b_actions[idx])
                    logratio = newlogprob - b_logprobs[idx]
                    ratio = logratio.exp()

                    mb_adv = b_advantages[idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((newvalue - b_returns[idx]) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            if global_step % self.write_frequency == 0:
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
                self.writer.add_scalar("losses/sasr_shaped_reward_mean", last_shaped_mean, global_step)
                self.writer.add_scalar("losses/sasr_S_buffer_size", float(len(self.S_buffer)), global_step)
                self.writer.add_scalar("losses/sasr_F_buffer_size", float(len(self.F_buffer)), global_step)
                self.writer.add_scalar("losses/sasr_shaping_active", last_shaping_active, global_step)
                if not math.isnan(self._last_threshold):
                    self.writer.add_scalar("losses/sasr_success_threshold", self._last_threshold, global_step)
                if len(self._terminal_rewards) > 0:
                    self.writer.add_scalar(
                        "losses/sasr_terminal_reward_mean",
                        float(np.mean(self._terminal_rewards)),
                        global_step,
                    )
                if not math.isnan(self._last_density_gap):
                    self.writer.add_scalar(
                        "losses/sasr_density_gap", self._last_density_gap, global_step
                    )
                if not math.isnan(self._last_beta_concentration):
                    self.writer.add_scalar(
                        "losses/sasr_beta_concentration",
                        self._last_beta_concentration,
                        global_step,
                    )

            pbar.set_postfix({
                "step": global_step,
                "shaped": "{:.3f}".format(last_shaped_mean),
                "active": int(last_shaping_active),
            })

            if global_step >= total_timesteps:
                break

        self.env.close()
        self.writer.close()

    def save(self, indicator="final"):
        path = os.path.join(
            self.save_folder, "sasr-ppo-ac-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)
        )
        torch.save(self.ac.state_dict(), path)
