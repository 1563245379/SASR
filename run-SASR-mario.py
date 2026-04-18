"""
The script to run SASR algorithm on Super Mario Bros environments.
"""

import argparse

from SASR.SASRAlgoDiscrete import SASRDiscrete
from SASR.Networks import SACActorDiscrete, QNetworkDiscrete
from SASR.utils import mario_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run SASR on Super Mario Bros.")

    parser.add_argument("--exp-name", type=str, default="sasr-mario")

    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v3")
    parser.add_argument("--movement", type=str, default="simple",
                        choices=["simple", "right_only", "complex"],
                        help="Action set: simple (7), right_only (5), complex (12)")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha-autotune", type=bool, default=True)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)

    parser.add_argument("--target-network-frequency", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-frequency", type=int, default=2)

    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-starts", type=int, default=10000)

    parser.add_argument("--reward-weight", type=float, default=0.6)
    parser.add_argument("--kde-bandwidth", type=float, default=0.2)
    parser.add_argument("--kde-sample-burnin", type=int, default=1000)

    parser.add_argument("--rff-dim", type=int, default=1000)
    parser.add_argument("--retention-rate", type=float, default=0.1)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--print-frequency", type=int, default=1,
                        help="Print average return every N episodes (0 to disable)")
    parser.add_argument("--save-folder", type=str, default="./sasr-mario/")

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    env = mario_env_maker(env_id=args.env_id, seed=args.seed, render=args.render, movement=args.movement)

    print(f"Environment: {args.env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    agent = SASRDiscrete(
        env=env,
        actor_class=SACActorDiscrete,
        critic_class=QNetworkDiscrete,
        exp_name=args.exp_name,
        seed=args.seed,
        cuda=args.cuda,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        rb_optimize_memory=args.rb_optimize_memory,
        batch_size=args.batch_size,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        target_network_frequency=args.target_network_frequency,
        tau=args.tau,
        policy_frequency=args.policy_frequency,
        alpha=args.alpha,
        alpha_autotune=args.alpha_autotune,
        reward_weight=args.reward_weight,
        kde_bandwidth=args.kde_bandwidth,
        kde_sample_burn_in=args.kde_sample_burnin,
        rff_dim=args.rff_dim,
        retention_rate=args.retention_rate,
        write_frequency=args.write_frequency,
        save_folder=args.save_folder,
    )

    agent.learn(total_timesteps=args.total_timesteps, learning_starts=args.learning_starts, print_frequency=args.print_frequency)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
