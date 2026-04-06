"""
Evaluate a trained SASR agent on continuous control environments.
Supports: policy evaluation (deterministic), Q-network accuracy, optional rendering, and training curve plotting.
"""

import argparse
import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

from SASR.Networks import SACActor, QNetworkContinuousControl
from SASR.utils import continuous_control_env_maker, classic_control_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SASR agent.")

    parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--model-dir", type=str, default="./SASR/",
                        help="Directory containing saved model files.")
    parser.add_argument("--exp-name", type=str, default="sasr")
    parser.add_argument("--indicator", type=str, default="final",
                        help="Model checkpoint indicator (e.g., 'final', 'best').")

    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation.")

    parser.add_argument("--plot-training", action="store_true",
                        help="Plot training curves from TensorBoard event files in runs/.")
    parser.add_argument("--runs-dir", type=str, default="./runs/",
                        help="Directory containing TensorBoard run folders.")

    parser.add_argument("--output-dir", type=str, default="./eval_results/",
                        help="Directory to save evaluation outputs (plots, etc.).")

    return parser.parse_args()


def load_models(env, args, device):
    """Load actor and Q-networks from saved checkpoints."""
    actor = SACActor(env).to(device)
    qf_1 = QNetworkContinuousControl(env).to(device)
    qf_2 = QNetworkContinuousControl(env).to(device)

    model_dir = args.model_dir
    exp_name = args.exp_name
    indicator = args.indicator
    seed = args.seed

    actor_path = os.path.join(model_dir, f"actor-{exp_name}-{indicator}-{seed}.pth")
    qf1_path = os.path.join(model_dir, f"qf_1-{exp_name}-{indicator}-{seed}.pth")
    qf2_path = os.path.join(model_dir, f"qf_2-{exp_name}-{indicator}-{seed}.pth")

    for path, name in [(actor_path, "Actor"), (qf1_path, "QF1"), (qf2_path, "QF2")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
    qf_1.load_state_dict(torch.load(qf1_path, map_location=device, weights_only=True))
    qf_2.load_state_dict(torch.load(qf2_path, map_location=device, weights_only=True))

    actor.eval()
    qf_1.eval()
    qf_2.eval()

    print(f"Loaded models from {model_dir} (exp={exp_name}, indicator={indicator}, seed={seed})")
    return actor, qf_1, qf_2


def evaluate_policy(env, actor, qf_1, qf_2, device, num_episodes, gamma):
    """
    Run deterministic policy evaluation.
    Returns per-episode stats: undiscounted return, discounted return, length, success,
    and initial Q-value prediction.
    """
    episode_returns = []
    episode_discounted_returns = []
    episode_lengths = []
    episode_successes = []
    q_predictions = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        discounted_return = 0.0
        step = 0
        discount = 1.0
        success = False
        initial_q = None

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                _, _, mean_action = actor.get_action(obs_tensor)

                # Record Q-value at the first step of each episode
                if step == 0:
                    q1_val = qf_1(obs_tensor, mean_action).item()
                    q2_val = qf_2(obs_tensor, mean_action).item()
                    initial_q = min(q1_val, q2_val)

            action = mean_action.cpu().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            discounted_return += discount * reward
            discount *= gamma
            step += 1

            # MountainCarContinuous-v0: success if position >= 0.45
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'state'):
                if env.unwrapped.state[0] >= 0.45:
                    success = True

        episode_returns.append(total_reward)
        episode_discounted_returns.append(discounted_return)
        episode_lengths.append(step)
        episode_successes.append(success)
        q_predictions.append(initial_q)

    return {
        "returns": np.array(episode_returns),
        "discounted_returns": np.array(episode_discounted_returns),
        "lengths": np.array(episode_lengths),
        "successes": np.array(episode_successes),
        "q_predictions": np.array(q_predictions),
    }


def print_stats(results):
    """Print evaluation statistics to terminal."""
    returns = results["returns"]
    disc_returns = results["discounted_returns"]
    lengths = results["lengths"]
    successes = results["successes"]
    q_preds = results["q_predictions"]

    print("\n" + "=" * 60)
    print("  POLICY EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:              {len(returns)}")
    print(f"  Mean Return:           {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"  Max Return:            {returns.max():.2f}")
    print(f"  Min Return:            {returns.min():.2f}")
    print(f"  Mean Discounted Return:{disc_returns.mean():.2f} ± {disc_returns.std():.2f}")
    print(f"  Mean Episode Length:   {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"  Success Rate:          {successes.mean() * 100:.1f}%")
    print("-" * 60)
    print("  Q-NETWORK EVALUATION")
    print("-" * 60)

    # Q prediction vs actual discounted return
    mae = np.mean(np.abs(q_preds - disc_returns))
    if len(q_preds) > 1 and np.std(q_preds) > 0 and np.std(disc_returns) > 0:
        correlation = np.corrcoef(q_preds, disc_returns)[0, 1]
    else:
        correlation = float('nan')

    print(f"  Mean Q Prediction:     {q_preds.mean():.2f} ± {q_preds.std():.2f}")
    print(f"  Mean Actual Disc. Ret: {disc_returns.mean():.2f} ± {disc_returns.std():.2f}")
    print(f"  MAE (Q vs Actual):     {mae:.2f}")
    print(f"  Correlation:           {correlation:.4f}")
    print("=" * 60 + "\n")


def plot_q_evaluation(results, output_dir):
    """Scatter plot of Q-value predictions vs actual discounted returns."""
    q_preds = results["q_predictions"]
    disc_returns = results["discounted_returns"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(disc_returns, q_preds, alpha=0.6, edgecolors='k', linewidths=0.5, s=50)

    # Diagonal reference line
    all_vals = np.concatenate([q_preds, disc_returns])
    lo, hi = all_vals.min() - 1, all_vals.max() + 1
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel("Actual Discounted Return", fontsize=12)
    ax.set_ylabel("Q-Value Prediction (min(Q1, Q2))", fontsize=12)
    ax.set_title("Q-Network Evaluation: Predicted vs Actual", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    mae = np.mean(np.abs(q_preds - disc_returns))
    if len(q_preds) > 1 and np.std(q_preds) > 0 and np.std(disc_returns) > 0:
        corr = np.corrcoef(q_preds, disc_returns)[0, 1]
    else:
        corr = float('nan')
    ax.text(0.05, 0.95, f"MAE = {mae:.2f}\nCorr = {corr:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "q_evaluation.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Q-value evaluation plot saved to {save_path}")


def plot_training_curves(runs_dir, output_dir):
    """Read TensorBoard event files and plot training curves."""
    try:
        from tbparse import SummaryReader
    except ImportError:
        print("tbparse not installed. Install with: pip install tbparse")
        print("Skipping training curve plot.")
        return

    # Find all event files
    event_files = glob.glob(os.path.join(runs_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"No TensorBoard event files found in {runs_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for run_dir_path in sorted(set(os.path.dirname(f) for f in event_files)):
        run_name = os.path.basename(run_dir_path)
        reader = SummaryReader(run_dir_path)
        df = reader.scalars

        # Look for episodic return tags
        return_tags = [t for t in df["tag"].unique() if "return" in t.lower() or "reward" in t.lower()]
        if not return_tags:
            continue

        for tag in return_tags:
            tag_data = df[df["tag"] == tag].sort_values("step")
            ax.plot(tag_data["step"], tag_data["value"], alpha=0.8, label=f"{run_name}/{tag}")

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Episodic Return", fontsize=12)
    ax.set_title("Training Curves", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves plot saved to {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment (same logic as run-SASR.py)
    env = (continuous_control_env_maker(env_id=args.env_id, seed=args.seed, render=args.render)
           if args.env_id.startswith("My")
           else classic_control_env_maker(env_id=args.env_id, seed=args.seed, render=args.render))

    # Load models
    actor, qf_1, qf_2 = load_models(env, args, device)

    # Evaluate
    print(f"\nRunning {args.num_episodes} evaluation episodes on {args.env_id}...")
    results = evaluate_policy(env, actor, qf_1, qf_2, device, args.num_episodes, args.gamma)

    # Print stats
    print_stats(results)

    # Plot Q-value evaluation
    plot_q_evaluation(results, args.output_dir)

    # Optionally plot training curves
    if args.plot_training:
        plot_training_curves(args.runs_dir, args.output_dir)

    env.close()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
