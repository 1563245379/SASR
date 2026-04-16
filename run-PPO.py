"""
Run PPO on continuous control environments (same env routing as run-SASR.py).
Default hyperparameters are biased toward sparse classic control (e.g. MountainCarContinuous-v0):
stronger entropy, larger rollouts, slightly higher GAE lambda, deeper MLP, and longer training than
the old generic defaults. Use explicit flags to reproduce older runs or other envs.
"""

import argparse

from SASR.PPOAlgo import PPO
from SASR.Networks import PPOActorCriticContinuous
from SASR.utils import continuous_control_env_maker, classic_control_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO on continuous control environments.")

    parser.add_argument("--exp-name", type=str, default="ppo")

    parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Train sequentially with seeds seed, seed+1, ... (separate checkpoints per seed).",
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.97,
        help="GAE lambda; closer to 1.0 stresses longer credit (often helps sparse episodic success).",
    )

    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.04,
        help="Entropy bonus; raised for sparse Mountain Car style exploration (try 0.02–0.08 if unstable).",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.75,
        help="Value loss coefficient (slightly higher than 0.5 default for harder value fitting).",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="Adam LR; slightly reduced vs 3e-4 when using higher ent_coef.",
    )

    parser.add_argument(
        "--steps-per-rollout",
        type=int,
        default=4096,
        help="On-policy batch size per update (larger lowers gradient variance; slower wall-clock per update).",
    )
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--no-anneal-lr", action="store_true", help="Keep learning rate constant.")

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total env steps per seed (Mountain Car often needs >1M for PPO; SASR paper used 1M for comparison).",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="MLP width for PPO actor-critic trunk.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=3,
        help="Number of ReLU hidden layers before actor/critic heads (old default was 2).",
    )
    parser.add_argument(
        "--actor-log-std-init",
        type=float,
        default=0.3,
        help="Initial value of learnable log-std parameter (before tanh squashing); higher => more initial exploration.",
    )

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-folder", type=str, default="./ppo/")

    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Per-step reward becomes r * scale + offset before learning (classic control only). Default 1 = unchanged MDP.",
    )
    parser.add_argument(
        "--reward-offset",
        type=float,
        default=0.0,
        help="Added after scaling. Compare to SASR only if SASR uses the same wrapper (it does not by default).",
    )

    return parser.parse_args()


def run():
    args = parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")

    def make_ac(env):
        return PPOActorCriticContinuous(
            env,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            actor_log_std_init=args.actor_log_std_init,
        )

    for k in range(args.num_seeds):
        run_seed = args.seed + k

        if args.env_id.startswith("My"):
            if args.reward_scale != 1.0 or args.reward_offset != 0.0:
                raise ValueError("--reward-scale/--reward-offset are only wired for classic env-id (not My*).")
            env = continuous_control_env_maker(env_id=args.env_id, seed=run_seed, render=args.render)
        else:
            env = classic_control_env_maker(
                env_id=args.env_id,
                seed=run_seed,
                render=args.render,
                reward_scale=args.reward_scale,
                reward_offset=args.reward_offset,
            )
            if args.reward_scale != 1.0 or args.reward_offset != 0.0:
                print(
                    "Reward shaping: r' = r * {:.4f} + {:.4f} (document this if comparing to SASR).".format(
                        args.reward_scale, args.reward_offset
                    )
                )

        print(
            "PPO train seed={} ({}/{}): ent_coef={} gae_lambda={} lr={} steps/rollout={} total={}".format(
                run_seed,
                k + 1,
                args.num_seeds,
                args.ent_coef,
                args.gae_lambda,
                args.learning_rate,
                args.steps_per_rollout,
                args.total_timesteps,
            )
        )

        agent = PPO(
            env=env,
            actor_critic_class=make_ac,
            exp_name=args.exp_name,
            seed=run_seed,
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
