"""
Train a Dueling DQN agent on Super Mario Bros using SASR environment wrappers.
Outputs TensorBoard logs to ./runs/ for comparison with SASR training curves.
Saves models in SASR-compatible format (actor + qf_1 + qf_2).
"""

import argparse
import datetime
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from SASR.utils import mario_env_maker


# ============================================================
# Dueling DQN Network (same architecture as Super-Mario-RL)
# ============================================================

class DuelingDQN(nn.Module):
    """Dueling DQN with shared CNN backbone identical to Super-Mario-RL/duel_dqn.py."""

    def __init__(self, n_frame, n_action, device):
        super().__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)   # advantage branch
        self.v = nn.Linear(512, 1)          # value branch
        self.device = device

        self._init_weights()

    def _init_weights(self):
        for m in [self.layer1, self.layer2]:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - adv.mean(dim=-1, keepdim=True))
        return q


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Training
# ============================================================

def train_step(q, q_target, memory, batch_size, gamma, optimizer, device):
    """Double DQN training step."""
    batch = memory.sample(batch_size)
    s, r, a, s_prime, done_mask = zip(*batch)

    s = np.array(s, dtype=np.float32).squeeze(1)        # (B, 4, 84, 84)
    s_prime = np.array(s_prime, dtype=np.float32).squeeze(1)

    s_t = torch.FloatTensor(s).to(device)
    s_prime_t = torch.FloatTensor(s_prime).to(device)
    r_t = torch.FloatTensor(r).unsqueeze(-1).to(device)
    a_t = torch.LongTensor(a).unsqueeze(-1).to(device)
    done_t = torch.FloatTensor(done_mask).unsqueeze(-1).to(device)

    # Double DQN: select action with q, evaluate with q_target
    with torch.no_grad():
        a_max = q(s_prime_t).argmax(dim=1, keepdim=True)
        y = r_t + gamma * q_target(s_prime_t).gather(1, a_max) * done_t

    q_value = q(s_t).gather(1, a_t)
    loss = F.smooth_l1_loss(q_value, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), q_value.mean().item()


def convert_and_save_sasr_format(q_net, n_action, save_folder, exp_name, indicator, seed):
    """Convert Dueling DQN weights to SASR-compatible actor + qf_1 + qf_2 format."""
    src = q_net.state_dict()

    # CNN backbone key mapping: DQN -> SASR CNNFeatureExtractor
    cnn_mapping = {
        'layer1.weight': 'feature_extractor.conv.0.weight',
        'layer1.bias': 'feature_extractor.conv.0.bias',
        'layer2.weight': 'feature_extractor.conv.2.weight',
        'layer2.bias': 'feature_extractor.conv.2.bias',
        'fc.weight': 'feature_extractor.fc.weight',
        'fc.bias': 'feature_extractor.fc.bias',
    }

    # --- Actor: advantage branch -> fc_logits (argmax(adv) == argmax(Q)) ---
    actor_sd = {}
    for dqn_key, sasr_key in cnn_mapping.items():
        actor_sd[sasr_key] = src[dqn_key].clone()
    actor_sd['fc_logits.weight'] = src['q.weight'].clone()
    actor_sd['fc_logits.bias'] = src['q.bias'].clone()

    # --- QNetwork: merge v + adv branches into single linear ---
    # Q(s,a) = v(s) + adv(s,a) - mean(adv(s,:))
    # fc_q.weight[a] = v.weight + q.weight[a] - mean(q.weight, dim=0)
    # fc_q.bias[a]   = v.bias   + q.bias[a]   - mean(q.bias)
    adv_w = src['q.weight']      # (n_action, 512)
    adv_b = src['q.bias']        # (n_action,)
    v_w = src['v.weight']        # (1, 512)
    v_b = src['v.bias']          # (1,)

    q_fc_weight = v_w.expand(n_action, -1) + adv_w - adv_w.mean(dim=0, keepdim=True)
    q_fc_bias = v_b.expand(n_action) + adv_b - adv_b.mean()

    qf_sd = {}
    for dqn_key, sasr_key in cnn_mapping.items():
        qf_sd[sasr_key] = src[dqn_key].clone()
    qf_sd['fc_q.weight'] = q_fc_weight
    qf_sd['fc_q.bias'] = q_fc_bias

    os.makedirs(save_folder, exist_ok=True)

    actor_path = os.path.join(save_folder, f"actor-{exp_name}-{indicator}-{seed}.pth")
    qf1_path = os.path.join(save_folder, f"qf_1-{exp_name}-{indicator}-{seed}.pth")
    qf2_path = os.path.join(save_folder, f"qf_2-{exp_name}-{indicator}-{seed}.pth")

    torch.save(actor_sd, actor_path)
    torch.save(qf_sd, qf1_path)
    torch.save(qf_sd, qf2_path)  # qf_1 == qf_2 (single Q network in DQN)

    print(f"Saved SASR-format models to {save_folder}:")
    print(f"  {actor_path}")
    print(f"  {qf1_path}")
    print(f"  {qf2_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dueling DQN on Mario (SASR wrappers)")

    parser.add_argument("--exp-name", type=str, default="duel-dqn-mario")
    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v1")
    parser.add_argument("--movement", type=str, default="right_only",
                        choices=["simple", "right_only", "complex"])
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.001,
                        help="Exploration epsilon (fixed)")

    parser.add_argument("--target-update-interval", type=int, default=50,
                        help="Copy weights to target network every N training steps")
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--total-episodes", type=int, default=10000)

    parser.add_argument("--print-frequency", type=int, default=10)
    parser.add_argument("--write-frequency", type=int, default=1,
                        help="Write to TensorBoard every N episodes")
    parser.add_argument("--save-frequency", type=int, default=500,
                        help="Save model every N episodes")
    parser.add_argument("--save-folder", type=str, default="./sasr-mario/")

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Environment (SASR wrappers)
    env = mario_env_maker(
        env_id=args.env_id, seed=args.seed, render=args.render, movement=args.movement
    )
    n_action = env.action_space.n
    print(f"Environment: {args.env_id}, actions: {n_action}, movement: {args.movement}")
    print(f"Observation space: {env.observation_space}")

    # Networks
    q = DuelingDQN(4, n_action, device).to(device)
    q_target = DuelingDQN(4, n_action, device).to(device)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=args.lr)

    # Replay buffer
    memory = ReplayBuffer(args.buffer_size)

    # TensorBoard
    run_name = "{}-{}-{}-{}".format(
        args.exp_name, "mario", args.seed,
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
    os.makedirs("./runs/", exist_ok=True)
    writer = SummaryWriter(os.path.join("./runs/", run_name))
    print(f"TensorBoard: ./runs/{run_name}")

    # Training loop
    global_train_steps = 0
    total_score = 0.0
    total_loss = 0.0
    episode_count = 0
    start_time = time.perf_counter()

    for ep in range(args.total_episodes):
        obs, _ = env.reset()
        obs = obs[np.newaxis, ...]  # (1, 4, 84, 84)
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_loss_sum = 0.0
        ep_train_count = 0

        while not done:
            # Epsilon-greedy action selection
            if args.epsilon > np.random.rand():
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = q(torch.FloatTensor(obs).to(device))
                    action = q_vals.argmax(dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = next_obs[np.newaxis, ...]

            # Store transition (done_mask: 1 if not done, 0 if done)
            memory.push((obs, float(reward), int(action), next_obs, int(not done)))
            obs = next_obs

            ep_reward += reward
            ep_length += 1

            # Train
            if len(memory) > args.learning_starts:
                loss_val, q_val = train_step(
                    q, q_target, memory, args.batch_size, args.gamma, optimizer, device
                )
                ep_loss_sum += loss_val
                ep_train_count += 1
                global_train_steps += 1

                # Update target network
                if global_train_steps % args.target_update_interval == 0:
                    q_target.load_state_dict(q.state_dict())

        episode_count += 1
        total_score += ep_reward
        if ep_train_count > 0:
            total_loss += ep_loss_sum / ep_train_count

        # TensorBoard logging
        if episode_count % args.write_frequency == 0:
            writer.add_scalar("charts/episodic_return", ep_reward, episode_count)
            writer.add_scalar("charts/episodic_length", ep_length, episode_count)
            if ep_train_count > 0:
                writer.add_scalar("losses/q_loss", ep_loss_sum / ep_train_count, episode_count)

        # Print
        if episode_count % args.print_frequency == 0:
            elapsed = time.perf_counter() - start_time
            start_time = time.perf_counter()
            avg_score = total_score / args.print_frequency
            avg_loss = total_loss / args.print_frequency
            print(f"Episode {episode_count} | avg_return: {avg_score:.2f} | "
                  f"avg_loss: {avg_loss:.4f} | train_steps: {global_train_steps} | "
                  f"buffer: {len(memory)} | time: {elapsed:.1f}s")
            total_score = 0.0
            total_loss = 0.0

        # Save
        if episode_count % args.save_frequency == 0:
            convert_and_save_sasr_format(
                q_target, n_action, args.save_folder,
                args.exp_name, f"ep{episode_count}", args.seed
            )

    # Final save
    convert_and_save_sasr_format(
        q_target, n_action, args.save_folder,
        args.exp_name, "final", args.seed
    )

    writer.close()
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
