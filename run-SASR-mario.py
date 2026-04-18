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

    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v1")
    parser.add_argument("--movement", type=str, default="simple",
                        choices=["simple", "right_only", "complex"],
                        help="Action set: simple (7), right_only (5), complex (12)")
    parser.add_argument("--render", action="store_true", help="Whether to render the environment")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--rb-optimize-memory", action="store_true", help="Whether to optimize replay buffer memory usage")
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
    parser.add_argument("--learning-starts", type=int, default=5000)

    parser.add_argument("--reward-weight", type=float, default=0.6)
    parser.add_argument("--kde-bandwidth", type=float, default=0.2)
    parser.add_argument("--kde-sample-burnin", type=int, default=1000)

    parser.add_argument("--rff-dim", type=int, default=1000)
    parser.add_argument("--retention-rate", type=float, default=0.1)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--print-frequency", type=int, default=10,
                        help="Print average return every N episodes (0 to disable)")
    parser.add_argument("--save-folder", type=str, default="./sasr-mario/")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true", default=False,
                        help="Enable curriculum learning (start near goal, progress to start)")
    parser.add_argument("--min-stage-episodes", type=int, default=200,
                        help="Minimum episodes per curriculum stage before evaluation starts")
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Evaluate every N episodes after min-stage-episodes reached")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes per check")
    parser.add_argument("--max-stage-episodes", type=int, default=1000,
                        help="Maximum episodes per stage (force advance to next stage)")
    parser.add_argument("--pass-rate-threshold", type=float, default=0.5,
                        help="Success rate threshold to advance to next stage")

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    # Curriculum positions: (x_pos, y_pos) from near-goal to near-start
    CURRICULUM_POSITIONS = [
        (2600, 79),
        (2000, 79),
        (1500, 79),
        (1000, 79),
        (500, 79),
    ]

    env = mario_env_maker(
        env_id=args.env_id, seed=args.seed, render=args.render, movement=args.movement,
        curriculum_positions=CURRICULUM_POSITIONS if args.curriculum else None,
    )

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

    if args.curriculum:
        agent.curriculum_learn(
            learning_starts=args.learning_starts,
            print_frequency=args.print_frequency,
            min_stage_episodes=args.min_stage_episodes,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            max_stage_episodes=args.max_stage_episodes,
            pass_rate_threshold=args.pass_rate_threshold,
        )
    else:
        agent.learn(total_timesteps=args.total_timesteps, learning_starts=args.learning_starts, print_frequency=args.print_frequency)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
