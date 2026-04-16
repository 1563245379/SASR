"""
Evaluate a trained PPO (discrete / CNN) agent on Super Mario Bros.
Loads checkpoint from run-PPO-mario.py: ppo-ac-{exp}-{indicator}-{seed}.pth
"""

import argparse
import os

import numpy as np
import torch

from SASR.Networks import PPOActorCriticDiscrete
from SASR.utils import mario_env_maker

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO-Mario agent.")

    parser.add_argument("--env-id", type=str, default="SuperMarioBros-v3")
    parser.add_argument(
        "--movement",
        type=str,
        default="simple",
        choices=["simple", "right_only", "complex"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument(
        "--model-dir",
        type=str,
        default="./ppo-mario/",
        help="Directory containing ppo-ac-*.pth",
    )
    parser.add_argument("--exp-name", type=str, default="ppo-mario")
    parser.add_argument("--indicator", type=str, default="final")

    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of greedy argmax.",
    )

    return parser.parse_args()


def load_model(env, args, device):
    ac = PPOActorCriticDiscrete(env).to(device)
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


def evaluate(env, ac, device, num_episodes, gamma, stochastic, render):
    returns = []
    disc_returns = []
    lengths = []
    flag_gets = []

    for ep in tqdm(range(num_episodes), desc="PPO-Mario eval"):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        disc_r = 0.0
        discount = 1.0
        steps = 0
        got_flag = False

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                if stochastic:
                    action, _, _, _ = ac.get_action_and_value(obs_tensor)
                    action = int(action.cpu().numpy().reshape(-1)[0])
                else:
                    action = int(ac.get_deterministic_action(obs_tensor).cpu().numpy().reshape(-1)[0])

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_r += reward
            disc_r += discount * reward
            discount *= gamma
            steps += 1

            if info.get("flag_get", False):
                got_flag = True

        returns.append(total_r)
        disc_returns.append(disc_r)
        lengths.append(steps)
        flag_gets.append(got_flag)

        if render:
            print(
                "Episode {}: return={:.4f}, len={}, flag_get={}".format(
                    ep + 1, total_r, steps, got_flag
                )
            )

    return {
        "returns": np.array(returns),
        "discounted_returns": np.array(disc_returns),
        "lengths": np.array(lengths),
        "flag_gets": np.array(flag_gets),
    }


def main():
    args = parse_args()
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    env = mario_env_maker(
        env_id=args.env_id,
        seed=args.seed,
        render=args.render,
        movement=args.movement,
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
    )

    r = results["returns"]
    dr = results["discounted_returns"]
    lens = results["lengths"]
    flags = results["flag_gets"]

    print("")
    print("=" * 60)
    print("  PPO-MARIO EVALUATION")
    print("=" * 60)
    print("  env:            {}".format(args.env_id))
    print("  episodes:       {}".format(len(r)))
    print(
        "  policy:         {}".format("stochastic" if args.stochastic else "deterministic (argmax)")
    )
    print("  mean return:    {:.4f}  (std {:.4f})".format(r.mean(), r.std()))
    print("  min / max:      {:.4f} / {:.4f}".format(r.min(), r.max()))
    print("  mean disc. ret: {:.4f}  (std {:.4f})".format(dr.mean(), dr.std()))
    print("  mean length:    {:.1f}  (std {:.1f})".format(lens.mean(), lens.std()))
    print("  flag_get rate:  {:.1f}%".format(100.0 * flags.mean()))
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
