import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler


class CryptoTradingEnv(gym.Env):
    """A basic cryptocurrency trading environment for reinforcement learning."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, data_path, window_size=20, initial_balance=10000, commission=0.001
    ):
        """Initialize the trading environment.

        Args:
            data_path (str): Path to the CSV file containing cryptocurrency data
            window_size (int): Number of previous observations to include in state
            initial_balance (float): Initial account balance
            commission (float): Trading commission rate (e.g., 0.001 for 0.1%)
        """
        super(CryptoTradingEnv, self).__init__()

        # Load and preprocess data
        self.df = pd.read_csv(data_path)
        self.df["date"] = pd.to_datetime(self.df["timestamp"])
        self.df.set_index("date", inplace=True)

        # Drop rows with missing values
        self.df.dropna(inplace=True)

        # Environment parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission

        # Features for the state
        self.features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "macd",
            "signal",
            "histogram",
        ]

        # Normalize features
        self.scaler = StandardScaler()
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

        # Action and observation spaces
        # Action: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)

        # Observation space: previous window_size of price data + current position
        obs_shape = (window_size + 1) * len(
            self.features
        ) + 2  # +2 for balance and holdings
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        # Episode variables (will be initialized in reset)
        self.current_step = None
        self.balance = None
        self.btc_held = None
        self.current_price = None
        self.account_history = None
        self.returns_history = None
        self.done = None

    def _next_observation(self):
        """Return the current state (observation)."""
        # Get the window of features
        frame = np.array([])

        # If not enough data for the window, pad with zeros
        if self.current_step < self.window_size:
            # For padding
            padding = np.zeros(
                (self.window_size - self.current_step) * len(self.features)
            )
            frame = np.append(
                padding, self.df[self.features].values[: self.current_step].flatten()
            )
        else:
            # Normal case
            frame = (
                self.df[self.features]
                .values[self.current_step - self.window_size : self.current_step]
                .flatten()
            )

        # Add the current price data
        current_data = self.df[self.features].values[self.current_step]
        frame = np.append(frame, current_data)

        # Append the account information: balance and BTC held
        frame = np.append(frame, [self.balance / self.initial_balance, self.btc_held])

        return frame

    def _calculate_reward(self, action):
        """Calculate the reward for the current step."""
        # Reward is the change in portfolio value
        prev_portfolio_value = (
            self.balance
            + self.btc_held * self.df["close"].values[max(0, self.current_step - 1)]
        )
        current_portfolio_value = self.balance + self.btc_held * self.current_price

        reward = current_portfolio_value - prev_portfolio_value

        # Penalize for trading (to account for commission and encourage less frequent trading)
        if action != 1:  # If action is not Hold
            reward -= self.commission * current_portfolio_value

        return reward

    def step(self, action):
        """Take an action in the environment and return the next state, reward, done, and info.

        Args:
            action (int): The action to take (0=Sell, 1=Hold, 2=Buy)

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Execute the action
        self._take_action(action)

        # Move to the next step
        self.current_step += 1

        # Check if we're at the end of the data
        if self.current_step >= len(self.df):
            self.done = True

        # Update current price
        if not self.done:
            self.current_price = self.df["close"].values[self.current_step]

        # Calculate reward
        reward = self._calculate_reward(action)

        # Get the next observation
        obs = self._next_observation()

        # Update account history
        portfolio_value = self.balance + self.btc_held * self.current_price
        self.account_history.append(portfolio_value)
        self.returns_history.append((portfolio_value / self.initial_balance) - 1)

        # Return step information
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "btc_held": self.btc_held,
            "portfolio_value": portfolio_value,
            "return": self.returns_history[-1],
        }

        return obs, reward, self.done, info

    def _take_action(self, action):
        """Execute the specified action.

        Args:
            action (int): The action to take (0=Sell, 1=Hold, 2=Buy)
        """
        # Get current BTC price
        self.current_price = self.df["close"].values[self.current_step]

        # Calculate transaction amount (fixed for simplicity)
        transaction_amount = 1.0  # Buy/sell 1 BTC at a time

        # Apply commission for transactions
        commission_amount = self.commission * self.current_price * transaction_amount

        if action == 0:  # Sell
            # Can only sell if we have BTC
            if self.btc_held > 0:
                # Sell at most what we have
                sell_amount = min(self.btc_held, transaction_amount)
                self.balance += sell_amount * self.current_price * (1 - self.commission)
                self.btc_held -= sell_amount

        elif action == 2:  # Buy
            # Can only buy if we have enough balance
            max_btc_can_buy = self.balance / (
                self.current_price * (1 + self.commission)
            )

            if max_btc_can_buy > 0:
                buy_amount = min(transaction_amount, max_btc_can_buy)
                self.balance -= buy_amount * self.current_price * (1 + self.commission)
                self.btc_held += buy_amount

        # Action 1 (Hold) does nothing

    def reset(self):
        """Reset the environment to its initial state.

        Returns:
            numpy.array: The initial observation
        """
        # Reset episode variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.current_price = self.df["close"].values[self.current_step]
        self.account_history = [self.initial_balance]
        self.returns_history = [0]
        self.done = False

        return self._next_observation()

    def render(self, mode="human", close=False):
        """Render the environment."""
        if mode == "human":
            portfolio_value = self.balance + self.btc_held * self.current_price
            print(f"Step: {self.current_step}")
            print(f"Price: ${self.current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"BTC Held: {self.btc_held:.4f}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(
                f"Return: {((portfolio_value / self.initial_balance) - 1) * 100:.2f}%"
            )
            print("-" * 50)

    def plot_results(self):
        """Plot the results of the episode."""
        plt.figure(figsize=(12, 8))

        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(self.account_history)
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Step")
        plt.ylabel("Portfolio Value ($)")

        # Plot returns
        plt.subplot(2, 1, 2)
        plt.plot([r * 100 for r in self.returns_history])
        plt.title("Returns Over Time")
        plt.xlabel("Step")
        plt.ylabel("Return (%)")

        plt.tight_layout()
        plt.show()

    def close(self):
        """Close the environment."""
        pass
