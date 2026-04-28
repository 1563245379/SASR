"""
Run PPO with SASR shaped-reward on Super Mario Bros (image observations, discrete actions).
Uses the same env preprocessing as run-PPO-mario.py / run-SASR-mario.py (mario_env_maker).
"""

import argparse

from SASR.SASRPPOAlgoDiscrete import SASRPPODiscrete
from SASR.Networks import PPOActorCriticDiscrete
from SASR.utils import mario_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO + SASR shaping on Super Mario Bros.")

    parser.add_argument("--exp-name", type=str, default="sasr-ppo-mario")

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

    # PPO hyperparameters (mirror run-PPO-mario.py)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus for discrete policy.")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)

    parser.add_argument("--steps-per-rollout", type=int, default=2048)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--no-anneal-lr", action="store_true", help="Keep learning rate constant.")

    parser.add_argument("--total-timesteps", type=int, default=1_050_000)

    parser.add_argument("--write-frequency", type=int, default=50)
    parser.add_argument("--save-folder", type=str, default="./sasr-ppo-mario/")

    # SASR shaping hyperparameters (mirror run-SASR-mario.py, plus a buffer cap)
    parser.add_argument("--reward-weight", type=float, default=0.6,
                        help="Weight on the SASR Beta-sampled shaped reward (halved internally). "
                             "If the policy is collapsing to a fixed action, lower this (e.g. 0.1) "
                             "and/or raise --ent-coef (e.g. 0.02).")
    parser.add_argument("--kde-bandwidth", type=float, default=0.2)
    parser.add_argument("--kde-sample-burnin", type=int, default=1000,
                        help="Buffer must exceed this size before KDE returns non-zero.")
    parser.add_argument("--rff-dim", type=int, default=1000,
                        help="Random Fourier Features dimension; pass <=0 to disable RFF and use exact KDE.")
    parser.add_argument("--retention-rate", type=float, default=0.1)
    parser.add_argument("--feature-dim", type=int, default=512,
                        help="CNN feature dim used for KDE (must match PPOActorCriticDiscrete.features output).")
    parser.add_argument("--buffer-max-size", type=int, default=50000,
                        help="Default FIFO cap applied to both S and F buffers (overridable below).")
    parser.add_argument("--s-buffer-max-size", type=int, default=None,
                        help="FIFO cap on the success (S) buffer. Defaults to --buffer-max-size. "
                             "Set lower (e.g. 5000) on hard tasks to keep S fresh and avoid "
                             "stale-elite lock-in: when CNN features and policy drift, old S "
                             "entries can drag the policy back toward early-training behavior.")
    parser.add_argument("--f-buffer-max-size", type=int, default=None,
                        help="FIFO cap on the failure (F) buffer. Defaults to --buffer-max-size. "
                             "Larger F is usually fine — failure modes are stable.")
    parser.add_argument("--shaping-warmup-steps", type=int, default=0,
                        help="Hard env-step floor before shaping engages, on top of the S-buffer "
                             "burn-in gate. Set e.g. 50000 if you want a guaranteed pure-PPO warmup "
                             "while the policy gathers initial successes.")
    parser.add_argument("--success-criterion", type=str, default="flag_or_quantile",
                        choices=["flag", "quantile", "flag_or_quantile"],
                        help="How to classify a finished trajectory as success (-> S buffer) vs "
                             "failure (-> F). 'flag': only info[flag_get] (reference SASR). "
                             "'quantile': S iff terminal reward >= rolling success-quantile of "
                             "recent terminal rewards (adaptive — never starves S). "
                             "'flag_or_quantile' (default): either condition. Use 'quantile' or "
                             "'flag_or_quantile' on hard levels where the agent rarely sees the flag.")
    parser.add_argument("--success-quantile", type=float, default=0.75,
                        help="Quantile threshold for adaptive success classification. 0.5 = "
                             "median split (~50/50 S/F); 0.75 = elite top-25%% feeds S.")
    parser.add_argument("--success-window", type=int, default=100,
                        help="Rolling window of recent terminal rewards used to compute the "
                             "adaptive threshold.")
    parser.add_argument("--no-feature-normalize", action="store_true",
                        help="Disable L2-normalization of CNN features before KDE. By default "
                             "features are normalized so kde_bandwidth has a stable cosine-"
                             "distance interpretation across training. Disable only to "
                             "reproduce the un-normalized reference SASR-Discrete behavior.")

    return parser.parse_args()


def run():
    args = parse_args()

    env = mario_env_maker(env_id=args.env_id, seed=args.seed, render=args.render, movement=args.movement)

    print("Environment: {}".format(args.env_id))
    print("Observation space: {}".format(env.observation_space))
    print("Action space: {}".format(env.action_space))

    rff_dim = args.rff_dim if args.rff_dim and args.rff_dim > 0 else None

    agent = SASRPPODiscrete(
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
        reward_weight=args.reward_weight,
        kde_bandwidth=args.kde_bandwidth,
        kde_sample_burn_in=args.kde_sample_burnin,
        rff_dim=rff_dim,
        retention_rate=args.retention_rate,
        feature_dim=args.feature_dim,
        buffer_max_size=args.buffer_max_size,
        shaping_warmup_steps=args.shaping_warmup_steps,
        success_criterion=args.success_criterion,
        success_quantile=args.success_quantile,
        success_window=args.success_window,
        feature_normalize=not args.no_feature_normalize,
        s_buffer_max_size=args.s_buffer_max_size,
        f_buffer_max_size=args.f_buffer_max_size,
    )

    agent.learn(total_timesteps=args.total_timesteps)
    agent.save(indicator="final")


if __name__ == "__main__":
    run()
