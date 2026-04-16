"""
Proximal Policy Optimization (PPO) for continuous control, single-environment rollout.
Reference style: cleanrl PPO continuous (GAE-Lambda, clipped surrogate).
"""

import os
import datetime
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


class PPO:
    def __init__(
        self,
        env,
        actor_critic_class,
        exp_name="ppo",
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
        save_folder="./ppo/",
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

        run_name = "{}-{}-{}-{}".format(
            exp_name,
            env.unwrapped.spec.id,
            seed,
            datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S"),
        )
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))

        os.makedirs(self.save_folder, exist_ok=True)

    def learn(self, total_timesteps=1_000_000):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))

        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        global_step = 0
        num_updates = max(1, int(math.ceil(total_timesteps / self.steps_per_rollout)))
        pbar = tqdm(range(num_updates), desc="PPO Learning")

        for update in pbar:
            if self.anneal_lr:
                frac = 1.0 - (update / max(1, num_updates - 1))
                self.optimizer.param_groups[0]["lr"] = frac * self.initial_lr

            mb_obs = np.zeros((self.steps_per_rollout, obs_dim), dtype=np.float32)
            mb_actions = np.zeros((self.steps_per_rollout, act_dim), dtype=np.float32)
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
                mb_actions[step] = action.cpu().numpy().reshape(-1)
                mb_logprobs[step] = float(logprob.cpu().numpy().reshape(-1)[0])
                mb_values[step] = float(value.cpu().numpy().reshape(-1)[0])

                next_obs, reward, terminated, truncated, info = self.env.step(
                    mb_actions[step].reshape(self.env.action_space.shape)
                )
                done = float(terminated or truncated)

                if "episode" in info:
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

                mb_rewards[step] = reward
                mb_dones[step] = done

                obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                if done:
                    obs, _ = self.env.reset()
                    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

            with torch.no_grad():
                next_v = self.ac.get_action_and_value(torch.as_tensor(obs, device=self.device).unsqueeze(0))[3]
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
            b_actions = torch.as_tensor(mb_actions, device=self.device)
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

            pbar.set_postfix({"step": global_step})

            if global_step >= total_timesteps:
                break

        self.env.close()
        self.writer.close()

    def save(self, indicator="final"):
        path = os.path.join(self.save_folder, "ppo-ac-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed))
        torch.save(self.ac.state_dict(), path)
