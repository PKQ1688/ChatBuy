import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.env_base_trad import CryptoTradingEnv

# Configuration
DATA_PATH = "../data/BTC_USDT_1d_with_indicators.csv"
WINDOW_SIZE = 20
TRAIN_TEST_SPLIT = 0.8  # 80% of data for training, 20% for testing
TOTAL_TIMESTEPS = 100000
RANDOM_SEED = 42
EVAL_FREQ = 10000


def main():
    """Main training function."""
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Calculate split index
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)

    # Create train and test dataframes
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Save train and test datasets temporarily
    train_path = "../data/train_data_temp.csv"
    test_path = "../data/test_data_temp.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Create and wrap the training environment
    env = CryptoTradingEnv(train_path, window_size=WINDOW_SIZE)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Create evaluation environment
    eval_env = CryptoTradingEnv(test_path, window_size=WINDOW_SIZE)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    # Initialize the agent (PPO algorithm)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/",
        create_eval_env=False,
        policy_kwargs=None,
        verbose=1,
        seed=RANDOM_SEED,
        device="auto",
        _init_setup_model=True,
    )

    # Train the agent
    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, log_interval=10
    )

    # Save the trained model
    model.save("crypto_trading_ppo")

    # Evaluate the model
    print("Evaluating the model...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Run a test episode for visualization
    print("Running a test episode...")
    run_test_episode(model, test_path)

    # Clean up temporary files
    os.remove(train_path)
    os.remove(test_path)


def run_test_episode(model, test_data_path):
    """Run a test episode using the trained model and visualize the results.

    Args:
        model: The trained RL model
        test_data_path: Path to the test data
    """
    # Create a test environment
    env = CryptoTradingEnv(test_data_path, window_size=WINDOW_SIZE)

    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0

    # Store actions for visualization
    actions = []

    # Run through the test data
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        actions.append(action)

        # Optionally render the environment
        # env.render()

    # Plot results
    env.plot_results()

    # Plot actions (Buy, Hold, Sell)
    plt.figure(figsize=(12, 6))
    plt.plot(actions)
    plt.title("Actions Taken by Agent (0=Sell, 1=Hold, 2=Buy)")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.yticks([0, 1, 2], ["Sell", "Hold", "Buy"])
    plt.grid(True)
    plt.show()

    # Print final results
    portfolio_value = env.balance + env.btc_held * env.current_price
    print(f"Final portfolio value: ${portfolio_value:.2f}")
    print(f"Return: {((portfolio_value / env.initial_balance) - 1) * 100:.2f}%")
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
