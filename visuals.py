import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque

# Ensure the plots directory exists
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_reward(global_steps_ep_end, episodic_rewards, smooth_window=100, save_path=os.path.join(PLOTS_DIR, "episodic_reward.png")):
    """
    Plots episodic rewards (raw and smoothed).

    Args:
        global_steps_ep_end: List of global steps where episodes ended.
        episodic_rewards: List of rewards for each finished episode.
        smooth_window: Window size for the rolling average.
        save_path: Path to save the plot image.
    """
    if not episodic_rewards:
        print("No episodic rewards recorded, skipping reward plot.")
        return

    plt.figure(figsize=(10, 5))

    # Calculate rolling average
    rewards_deque = deque(maxlen=smooth_window)
    smoothed_rewards = []
    for rew in episodic_rewards:
        rewards_deque.append(rew)
        smoothed_rewards.append(np.mean(rewards_deque))

    # Plot raw rewards (lightly)
    plt.plot(global_steps_ep_end, episodic_rewards, color='lightblue', linewidth=0.5, alpha=1.0, label='Raw Episodic Reward')

    # Plot smoothed rewards (prominently)
    plt.plot(global_steps_ep_end, smoothed_rewards, color='darkblue', linewidth=1.5, label=f'Smoothed Reward (Window {smooth_window})')

    plt.title("Episodic Reward over Time")
    plt.xlabel("Global Timestep")
    plt.ylabel("Episodic Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Reward plot saved to {save_path}")


def plot_losses(update_steps, policy_losses, value_losses, entropy_losses, total_losses, save_path=os.path.join(PLOTS_DIR, "losses.png")):
    """
    Plots policy, value, entropy, and total losses on a 2x2 grid.

    Args:
        update_steps: List of update steps.
        policy_losses: List of policy losses per update.
        value_losses: List of value losses per update.
        entropy_losses: List of entropy losses per update.
        total_losses: List of total losses per update.
        save_path: Path to save the plot image.
    """
    if not update_steps:
        print("No updates recorded, skipping loss plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

    axs[0, 0].plot(update_steps, policy_losses)
    axs[0, 0].set_title("Policy Loss")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True)

    axs[0, 1].plot(update_steps, value_losses)
    axs[0, 1].set_title("Value Loss")
    axs[0, 1].grid(True)

    axs[1, 0].plot(update_steps, entropy_losses)
    axs[1, 0].set_title("Entropy Loss")
    axs[1, 0].set_xlabel("Update Step")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True)

    axs[1, 1].plot(update_steps, total_losses)
    axs[1, 1].set_title("Total Loss")
    axs[1, 1].set_xlabel("Update Step")
    axs[1, 1].grid(True)

    fig.suptitle("Training Losses")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def plot_diagnostics(update_steps, explained_vars, clip_fracs, learning_rates, save_path=os.path.join(PLOTS_DIR, "diagnostics.png")):
    """
    Plots explained variance, clip fraction, and learning rate.

    Args:
        update_steps: List of update steps.
        explained_vars: List of explained variances per update.
        clip_fracs: List of average clip fractions per update.
        learning_rates: List of learning rates per update.
        save_path: Path to save the plot image.
    """
    if not update_steps:
        print("No updates recorded, skipping diagnostics plot.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    axs[0].plot(update_steps, explained_vars)
    axs[0].set_title("Explained Variance")
    axs[0].set_xlabel("Update Step")
    axs[0].set_ylabel("EV")
    axs[0].grid(True)

    axs[1].plot(update_steps, clip_fracs)
    axs[1].set_title("Clip Fraction")
    axs[1].set_xlabel("Update Step")
    axs[1].set_ylabel("Fraction Clipped")
    axs[1].grid(True)

    axs[2].plot(update_steps, learning_rates)
    axs[2].set_title("Learning Rate")
    axs[2].set_xlabel("Update Step")
    axs[2].set_ylabel("LR")
    axs[2].grid(True)

    fig.suptitle("Training Diagnostics")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.savefig(save_path)
    plt.close()
    print(f"Diagnostics plot saved to {save_path}")


def save_data(global_steps_ep_end, episodic_rewards, update_steps, total_losses, save_path=os.path.join(PLOTS_DIR, "training_data.npz")):
    """
    Saves key training data arrays to a compressed NumPy file.

    Args:
        global_steps_ep_end: List of global steps where episodes ended.
        episodic_rewards: List of rewards for each finished episode.
        update_steps: List of update steps.
        total_losses: List of total losses per update.
        save_path: Path to save the .npz file.
    """
    try:
        np.savez(
            save_path,
            global_steps_ep_end=np.array(global_steps_ep_end),
            episodic_rewards=np.array(episodic_rewards),
            update_steps=np.array(update_steps),
            total_losses=np.array(total_losses)
        )
        print(f"Training data saved to {save_path}")
    except Exception as e:
        print(f"Error saving data to {save_path}: {e}")
