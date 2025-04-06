import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append("../")
from env.env_base_trad import TradingEnv

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "BTC_USDT_1d_with_indicators.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Custom callback for saving best model
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Callback for saving the model when the best mean reward is achieved.

    :param check_freq: (int) Frequency of checking the mean reward (in number of steps).
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param verbose: (int) Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Calculate mean reward from last 100 episodes
            x, y = self.model.ep_info_buffer.get_mean_rewards_and_episode_lengths(-100)
            mean_reward = np.mean(x)

            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(
                    f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}"
                )

            # New best model, save the agent
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")

                self.model.save(os.path.join(self.save_path, "best_model"))

        return True


def create_env(data_path, window_size=20, initial_balance=10000, validation=False):
    """Create the trading environment."""
    # Load data
    df = pd.read_csv(data_path)

    # Split data into training and validation sets (80/20 split)
    if validation:
        df = df.iloc[int(len(df) * 0.8) :]  # Last 20% for validation
    else:
        df = df.iloc[: int(len(df) * 0.8)]  # First 80% for training

    # Save the split data temporarily
    temp_path = os.path.join(
        BASE_DIR,
        "data",
        "train_data_temp.csv" if not validation else "test_data_temp.csv",
    )
    df.to_csv(temp_path, index=False)

    # Create the environment
    env = TradingEnv(
        data_path=temp_path,
        window_size=window_size,
        initial_balance=initial_balance,
        commission=0.001,  # 0.1% trading fee
        reward_scaling=100.0,  # Scale rewards for better gradient flow
    )

    return env


def train_model(model_type="PPO", total_timesteps=100000):
    """Train the RL model on the BTC trading environment."""
    # Create environments
    train_env = create_env(DATA_PATH, validation=False)
    val_env = create_env(DATA_PATH, validation=True)

    # Wrap environments with Monitor for logging
    train_env = Monitor(train_env, os.path.join(LOG_DIR, "train"))
    val_env = Monitor(val_env, os.path.join(LOG_DIR, "val"))

    # Define model
    if model_type == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        )
    elif model_type == "A2C":
        model = A2C(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            ent_coef=0.01,
            tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        )
    elif model_type == "DQN":
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=1000000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            batch_size=64,
            gamma=0.99,
            tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=LOG_DIR)

    # Train the model
    print(f"Training {model_type} model for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model
    model_save_path = os.path.join(
        MODEL_DIR,
        f"btc_trading_{model_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, model_save_path


def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model."""
    # Reset the environment
    obs = env.reset()

    # Initialize variables for tracking
    episode_rewards = []
    episode_lengths = []
    portfolio_values = []
    current_reward = 0
    current_length = 0

    # Run for multiple episodes
    for _ in range(num_episodes):
        done = False
        obs = env.reset()

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, done, info = env.step(action)

            # Track rewards and length
            current_reward += reward
            current_length += 1

            # Track portfolio value
            portfolio_values.append(info["portfolio_value"])

            if done:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                current_reward = 0
                current_length = 0

    # Calculate evaluation metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")

    return episode_rewards, portfolio_values


def backtest_strategy(model, data_path, window_size=20, initial_balance=10000):
    """Backtest the trained RL strategy on historical data."""
    # Create environment for backtesting
    env = create_env(
        data_path,
        window_size=window_size,
        initial_balance=initial_balance,
        validation=True,
    )

    # Reset environment
    obs = env.reset()

    # Track results
    portfolio_values = [initial_balance]
    actions_taken = []
    price_history = []

    # Run through the entire dataset
    done = False
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Save price and action
        price_history.append(env.current_price)
        actions_taken.append(action)

        # Take step in environment
        obs, reward, done, info = env.step(action)

        # Track portfolio value
        portfolio_values.append(info["portfolio_value"])

    # Calculate performance metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100

    # Calculate daily returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    )  # Annualized

    # Calculate drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown) * 100

    # Print results
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Plot results
    plt.figure(figsize=(14, 7))

    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)

    # Plot price and actions
    plt.subplot(2, 1, 2)
    plt.plot(price_history, "k-", label="BTC Price")

    # Mark buy and sell points
    buy_indices = [i for i, a in enumerate(actions_taken) if a == 1]
    sell_indices = [i for i, a in enumerate(actions_taken) if a == 2]

    plt.scatter(
        buy_indices,
        [price_history[i] for i in buy_indices],
        marker="^",
        color="g",
        label="Buy",
    )
    plt.scatter(
        sell_indices,
        [price_history[i] for i in sell_indices],
        marker="v",
        color="r",
        label="Sell",
    )

    plt.title("BTC Price and Trading Actions")
    plt.xlabel("Trading Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "backtest_results.png"))
    plt.show()

    return {
        "portfolio_values": portfolio_values,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "actions": actions_taken,
        "prices": price_history,
    }


if __name__ == "__main__":
    # Choose model type: "PPO", "A2C", or "DQN"
    MODEL_TYPE = "PPO"

    # Train model
    model, model_path = train_model(model_type=MODEL_TYPE, total_timesteps=100000)

    # Backtest strategy
    results = backtest_strategy(model, DATA_PATH)

    # To load and evaluate a saved model:
    # loaded_model = PPO.load("/path/to/saved/model")
    # results = backtest_strategy(loaded_model, DATA_PATH)
