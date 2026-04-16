"""
Evaluate a trained PPO actor-critic on continuous control environments.
Loads checkpoint written by run-PPO.py (ppo-ac-{exp}-{indicator}-{seed}.pth).
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from SASR.Networks import PPOActorCriticContinuous
from SASR.utils import continuous_control_env_maker, classic_control_env_maker


def classic_control_env_minimal(env_id, seed=1, render=False, reward_scale=1.0, reward_offset=0.0):
    """Same dynamics as classic_control_env_maker but without RecordEpisodeStatistics."""
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if reward_scale != 1.0 or reward_offset != 0.0:
        rs = float(reward_scale)
        ro = float(reward_offset)
        env = gym.wrappers.TransformReward(env, lambda r, _rs=rs, _ro=ro: r * _rs + _ro)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")

    parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument(
        "--model-dir",
        type=str,
        default="./ppo/",
        help="Directory containing ppo-ac-*.pth",
    )
    parser.add_argument("--exp-name", type=str, default="ppo-mc")
    parser.add_argument("--indicator", type=str, default="final")

    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of deterministic mean.",
    )

    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="Print per-step actions for the first few episodes (see --trace-*) and per-episode action stats.",
    )
    parser.add_argument(
        "--trace-episodes",
        type=int,
        default=1,
        help="With --print-actions: number of episodes to print step-by-step (default: 1).",
    )
    parser.add_argument(
        "--trace-max-steps",
        type=int,
        default=50,
        help="With --print-actions: max steps printed per traced episode (default: 50).",
    )

    parser.add_argument(
        "--minimal-env",
        action="store_true",
        help="Use gym.make only + seed (no RecordEpisodeStatistics). Reward is unchanged; use to rule out wrapper confusion.",
    )

    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Linear reward scaling r' = r * scale + offset (must match training). Default 1.0 = Gymnasium reward.",
    )
    parser.add_argument(
        "--reward-offset",
        type=float,
        default=0.0,
        help="Added after scaling. Must match run-PPO.py if you trained with shaping.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Must match the network used in run-PPO.py for this checkpoint.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=3,
        help="Must match run-PPO.py (old checkpoints often used 2).",
    )

    return parser.parse_args()


def load_model(env, args, device):
    ac = PPOActorCriticContinuous(
        env,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)
    path = os.path.join(
        args.model_dir,
        "ppo-ac-{}-{}-{}.pth".format(args.exp_name, args.indicator, args.seed),
    )
    if not os.path.exists(path):
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    ac.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    ac.eval()
    print("Loaded {}".format(path))
    return ac


def mountaincar_success(env):
    """MountainCarContinuous-v0: car reaches right hill when position >= 0.45."""
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "state"):
        return bool(env.unwrapped.state[0] >= 0.45)
    return False


def _mc_position(env):
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "state"):
        return float(env.unwrapped.state[0])
    return None


def evaluate(
    env,
    ac,
    device,
    num_episodes,
    gamma,
    stochastic,
    render,
    print_actions=False,
    trace_episodes=1,
    trace_max_steps=50,
):
    returns = []
    disc_returns = []
    lengths = []
    successes = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        disc_r = 0.0
        discount = 1.0
        steps = 0
        success = False
        episode_actions = []

        while not done:
            obs_tensor = torch.as_tensor(
                np.asarray(obs, dtype=np.float32).reshape(1, -1),
                device=device,
            )

            with torch.no_grad():
                if stochastic:
                    action, _, _, _ = ac.get_action_and_value(obs_tensor)
                else:
                    action = ac.get_deterministic_action(obs_tensor)

            a = action.cpu().numpy().reshape(env.action_space.shape)
            episode_actions.append(np.asarray(a, dtype=np.float64).reshape(-1).copy())

            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            if print_actions and ep < trace_episodes and steps < trace_max_steps:
                pos = _mc_position(env)
                pos_s = "{:.4f}".format(pos) if pos is not None else "n/a"
                a_str = np.array2string(
                    np.asarray(a).reshape(-1), precision=4, suppress_small=True
                )
                print(
                    "  [trace ep={} t={}] action={} reward={:.5f} pos={} done={}".format(
                        ep + 1, steps, a_str, float(reward), pos_s, done
                    )
                )

            total_r += reward
            disc_r += discount * reward
            discount *= gamma
            steps += 1

            if mountaincar_success(env):
                success = True

        returns.append(total_r)
        disc_returns.append(disc_r)
        lengths.append(steps)
        successes.append(success)

        if print_actions and len(episode_actions) > 0:
            stacked = np.concatenate(episode_actions)
            uniq = len(np.unique(np.round(stacked, decimals=4)))
            print(
                "  [action stats ep={}] len={} mean={:.4f} std={:.4f} min={:.4f} max={:.4f} ~unique(4dp)={}".format(
                    ep + 1,
                    len(episode_actions),
                    float(stacked.mean()),
                    float(stacked.std()),
                    float(stacked.min()),
                    float(stacked.max()),
                    uniq,
                )
            )

        if render:
            print("Episode {}: return={:.2f}, len={}, success={}".format(ep + 1, total_r, steps, success))

    return {
        "returns": np.array(returns),
        "discounted_returns": np.array(disc_returns),
        "lengths": np.array(lengths),
        "successes": np.array(successes),
    }


def main():
    args = parse_args()
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    if args.env_id.startswith("My"):
        if args.minimal_env:
            raise ValueError("--minimal-env only applies to classic env-id (not My*).")
        env = continuous_control_env_maker(
            env_id=args.env_id, seed=args.seed, render=args.render
        )
    elif args.minimal_env:
        env = classic_control_env_minimal(
            env_id=args.env_id,
            seed=args.seed,
            render=args.render,
            reward_scale=args.reward_scale,
            reward_offset=args.reward_offset,
        )
    else:
        env = classic_control_env_maker(
            env_id=args.env_id,
            seed=args.seed,
            render=args.render,
            reward_scale=args.reward_scale,
            reward_offset=args.reward_offset,
        )

    if not args.env_id.startswith("My"):
        print(
            "Note: classic env reward = (Gymnasium r) * {:.4f} + {:.4f}; then RecordEpisodeStatistics (unless --minimal-env).".format(
                args.reward_scale, args.reward_offset
            )
        )
    else:
        print(
            "Note: My* env uses continuous_control_env_maker (includes TransformReward +1 for goal envs)."
        )

    ac = load_model(env, args, device)
    results = evaluate(
        env,
        ac,
        device,
        args.num_episodes,
        args.gamma,
        args.stochastic,
        args.render,
        print_actions=args.print_actions,
        trace_episodes=args.trace_episodes,
        trace_max_steps=args.trace_max_steps,
    )

    r = results["returns"]
    dr = results["discounted_returns"]
    lens = results["lengths"]
    succ = results["successes"]

    print("")
    print("=" * 60)
    print("  PPO EVALUATION")
    print("=" * 60)
    print("  env:            {}".format(args.env_id))
    print("  episodes:       {}".format(len(r)))
    print("  policy:         {}".format("stochastic" if args.stochastic else "deterministic (mean)"))
    print("  mean return:    {:.2f}  (std {:.2f})".format(r.mean(), r.std()))
    print("  min / max:      {:.2f} / {:.2f}".format(r.min(), r.max()))
    print("  mean disc. ret: {:.2f}  (std {:.2f})".format(dr.mean(), dr.std()))
    print("  mean length:    {:.1f}  (std {:.1f})".format(lens.mean(), lens.std()))
    print("  success rate:   {:.1f}%".format(100.0 * succ.mean()))
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
