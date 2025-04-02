import os

import matplotlib.pyplot as plt
import torch
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Import your custom environment
from chatbuy.rl_src.env.env_btc_ccxt import BitcoinEnv


def train_agent(
    data_cwd,
    price_ary=None,
    tech_ary=None,
    algorithm="PPO",
    total_timesteps=500000,
    eval_freq=10000,
    n_eval_episodes=5,
    save_path="./models/sb3_btc",
    log_path="./logs/sb3_btc",
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048,
    device="auto",
):
    """Train a reinforcement learning agent on Bitcoin trading environment.

    Args:
        data_cwd: Path to data directory
        price_ary: Price array (if not loading from data_cwd)
        tech_ary: Technical indicators array (if not loading from data_cwd)
        algorithm: RL algorithm to use ('PPO', 'A2C', 'SAC', or 'TD3')
        total_timesteps: Total timesteps for training
        eval_freq: Evaluation frequency in timesteps
        n_eval_episodes: Number of episodes for evaluation
        save_path: Path to save the trained model
        log_path: Path to save logs
        learning_rate: Learning rate for the algorithm
        batch_size: Batch size for training
        n_steps: Number of steps per update for on-policy algorithms
        device: Device to use ('auto', 'cpu', 'cuda')

    Returns:
        The trained model
    """
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Create training environment
    env_train = BitcoinEnv(
        data_cwd=data_cwd, price_ary=price_ary, tech_ary=tech_ary, mode="train"
    )

    # Create evaluation environment
    env_eval = BitcoinEnv(
        data_cwd=data_cwd, price_ary=price_ary, tech_ary=tech_ary, mode="test"
    )

    # Wrap environments with Monitor for logging
    env_train = Monitor(env_train, os.path.join(log_path, "train"))
    env_eval = Monitor(env_eval, os.path.join(log_path, "eval"))

    # Set up callbacks
    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=os.path.join(save_path, "best_model"),
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="btc_model",
    )

    # Initialize the appropriate algorithm
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env_train,
            learning_rate=learning_rate,
            n_steps=n_steps,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env_train,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
        )
    elif algorithm == "TD3":
        model = TD3(
            "MlpPolicy",
            env_train,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
        )
    else:
        raise ValueError(
            f"Algorithm {algorithm} not supported. Choose from 'PPO', 'A2C', 'SAC', or 'TD3'"
        )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback]
    )

    # Save the final model
    model.save(os.path.join(save_path, "final_model"))

    return model


def evaluate_agent(model, data_cwd=None, price_ary=None, tech_ary=None, mode="test"):
    """Evaluate a trained agent and plot performance."""
    # Create evaluation environment
    env = BitcoinEnv(
        data_cwd=data_cwd, price_ary=price_ary, tech_ary=tech_ary, mode=mode
    )

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=1, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Run through one full episode for plotting
    state = env.reset()
    done = False
    episode_returns = [1]  # Start with the initial value (1x initial account)
    btc_returns = []
    init_price = None

    while not done:
        if init_price is None:
            init_price = env.day_price[0]

        btc_returns.append(env.day_price[0] / init_price)
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        episode_returns.append(env.total_asset / env.initial_account)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(episode_returns, label="Agent Return")
    plt.plot(btc_returns, color="yellow", label="BTC Return")
    plt.grid(True)
    plt.title(f"Cumulative Return ({mode} mode)")
    plt.xlabel("Day")
    plt.ylabel("Multiple of Initial Account")
    plt.legend()

    # Save the plot
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/sb3_btc_{mode}_return.jpg")
    plt.close()

    return episode_returns, btc_returns


def main():
    # Set parameters
    data_cwd = (
        "/Users/zhutaonan/Desktop/chatbuy/chatbuy/rl_src/data"  # Adjust path as needed
    )
    algorithm = "PPO"  # Options: "PPO", "A2C", "SAC", "TD3"
    total_timesteps = 1000000

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train the agent
    model = train_agent(
        data_cwd=data_cwd,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        device=device,
    )

    # Evaluate the agent on test data
    evaluate_agent(model, data_cwd=data_cwd, mode="test")

    # Evaluate the agent on trading data
    evaluate_agent(model, data_cwd=data_cwd, mode="trade")

    print("Training and evaluation complete!")


if __name__ == "__main__":
    main()
