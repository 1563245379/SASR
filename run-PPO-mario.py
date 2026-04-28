"""
Run PPO on Super Mario Bros (image observations, discrete actions).
Uses the same env preprocessing as run-SASR-mario.py (mario_env_maker).
"""

import argparse

from SASR.PPOAlgoDiscrete import PPODiscrete
from SASR.Networks import PPOActorCriticDiscrete
from SASR.utils import mario_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO on Super Mario Bros.")

    parser.add_argument("--exp-name", type=str, default="ppo-mario")

    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v1")
    parser.add_argument(
        "--movement",
        type=str,
        default="simple",
        choices=["simple", "right_only", "complex"],
        help="Action set: simple (7), right_only (5), complex (12)",
    )
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy bonus for discrete policy.",
    )
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)

    parser.add_argument("--steps-per-rollout", type=int, default=2048)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--no-anneal-lr", action="store_true", help="Keep learning rate constant.")

    parser.add_argument("--total-timesteps", type=int, default=3_000_000)

    parser.add_argument("--write-frequency", type=int, default=50)
    parser.add_argument("--save-folder", type=str, default="./ppo-mario/")

    return parser.parse_args()


def run():
    args = parse_args()

    env = mario_env_maker(env_id=args.env_id, seed=args.seed, render=args.render, movement=args.movement)

    print("Environment: {}".format(args.env_id))
    print("Observation space: {}".format(env.observation_space))
    print("Action space: {}".format(env.action_space))

    agent = PPODiscrete(
        env=env,
        actor_critic_class=PPOActorCriticDiscrete,
        exp_name=args.exp_name,
        seed=args.seed,
        cuda=args.cuda,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        steps_per_rollout=args.steps_per_rollout,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        anneal_lr=not args.no_anneal_lr,
        write_frequency=args.write_frequency,
        save_folder=args.save_folder,
    )

    agent.learn(total_timesteps=args.total_timesteps)
    agent.save(indicator="final")


if __name__ == "__main__":
    run()
