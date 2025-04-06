from typing import Optional

import numpy as np
import pandas as pd


class TradingEnv:
    """A base class for cryptocurrency trading environment.

    This environment simulates trading of cryptocurrencies with features:
    - Discrete actions: Buy, Sell, Hold
    - Observations: Price data and technical indicators
    - Rewards: Returns based on portfolio value changes
    """

    def __init__(
        self,
        data_path: str,
        window_size: int = 20,
        initial_balance: float = 10000,
        commission: float = 0.001,
        reward_scaling: float = 1.0,
    ):
        """Initialize the trading environment.

        Args:
            data_path: Path to the CSV file with trading data
            window_size: Size of the observation window (lookback period)
            initial_balance: Initial account balance
            commission: Trading commission percentage (e.g., 0.001 = 0.1%)
            reward_scaling: Scaling factor for reward
        """
        self.data = self._load_data(data_path)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_scaling = reward_scaling

        # Define action and observation spaces
        self.action_space_size = 3  # Buy, Sell, Hold

        # Number of features (OHLCV + technical indicators)
        # timestamp, open, high, low, close, volume, bb_upper, bb_middle, bb_lower, macd, signal, histogram
        self.num_features = 11  # excluding timestamp

        # Size of flattened observation space
        self.observation_space_size = self.window_size * self.num_features

        # Initialize variables
        self.reset()

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess the data from CSV file."""
        df = pd.read_csv(data_path)

        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Fill NaN values in technical indicators
        # For this example, we'll forward fill, but you may want to use different strategies
        df = df.fillna(method="ffill")

        # Normalize the price and technical features to avoid large values
        price_columns = ["open", "high", "low", "close"]

        # Apply normalization to price columns relative to close
        for col in price_columns:
            df[f"{col}_norm"] = df[col] / df["close"].shift(1) - 1.0

        # Apply log normalization to volume
        df["volume_norm"] = np.log(df["volume"] + 1) / 15  # Arbitrary scaling

        # Normalize technical indicators
        indicator_columns = [
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "macd",
            "signal",
            "histogram",
        ]
        for col in indicator_columns:
            if col in ["bb_upper", "bb_middle", "bb_lower"]:
                df[f"{col}_norm"] = df[col] / df["close"].shift(1) - 1.0
            else:
                # For MACD-related features, they can be both positive and negative
                max_abs_val = max(abs(df[col].max()), abs(df[col].min()))
                if max_abs_val > 0:
                    df[f"{col}_norm"] = df[col] / max_abs_val
                else:
                    df[f"{col}_norm"] = df[col]

        # Drop rows with NaN (first few rows due to normalization)
        df = df.dropna()

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def reset(self):
        """Reset the environment to the initial state."""
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.current_step = (
            self.window_size
        )  # Start after window_size to have enough history

        # Track portfolio value history
        self.portfolio_values = [self.initial_balance]

        # Track positions for visualization later
        self.trades_done = []

        # Calculate initial portfolio value
        self.portfolio_value = self.balance + self.crypto_held * self.current_price

        return self._get_observation()

    @property
    def current_price(self) -> float:
        """Get the current closing price."""
        return self.data.iloc[self.current_step]["close"]

    def _get_observation(self) -> np.ndarray:
        """
        Get the observation at the current step.

        Returns:
            A flattened array of feature values from the current window
        """
        # Get window of data
        window_data = self.data.iloc[
            self.current_step - self.window_size : self.current_step
        ]

        # Extract normalized features
        feature_columns = [
            "open_norm",
            "high_norm",
            "low_norm",
            "close_norm",
            "volume_norm",
            "bb_upper_norm",
            "bb_middle_norm",
            "bb_lower_norm",
            "macd_norm",
            "signal_norm",
            "histogram_norm",
        ]

        # Create observation
        observation = window_data[feature_columns].values

        # Add account state features (normalized)
        # - Owned crypto as a fraction of what could be owned with the entire balance
        crypto_owned_norm = self.crypto_held * self.current_price / self.initial_balance

        # - Cash balance as a fraction of initial balance
        balance_norm = self.balance / self.initial_balance

        # Add these features to the observation
        observation_with_account = np.append(
            observation.flatten(), [crypto_owned_norm, balance_norm]
        )

        return observation_with_account

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take (0: Hold, 1: Buy, 2: Sell)

        Returns:
            observation: The next observation
            reward: The reward for taking the action
            done: Whether the episode is done
            info: Additional information
        """
        # Get current price
        current_price = self.current_price

        # Previous portfolio value for calculating returns
        prev_portfolio_value = self.portfolio_value

        # Execute action
        if action == 1:  # Buy
            # Calculate maximum crypto that can be bought
            max_crypto_to_buy = self.balance / (current_price * (1 + self.commission))

            # Buy all possible crypto
            self.crypto_held += max_crypto_to_buy
            self.balance -= max_crypto_to_buy * current_price * (1 + self.commission)

            # Record the trade
            self.trades_done.append(
                {
                    "step": self.current_step,
                    "type": "buy",
                    "price": current_price,
                    "amount": max_crypto_to_buy,
                    "cost": max_crypto_to_buy * current_price * (1 + self.commission),
                }
            )

        elif action == 2:  # Sell
            if self.crypto_held > 0:
                # Sell all crypto
                sell_amount = self.crypto_held
                self.balance += sell_amount * current_price * (1 - self.commission)
                self.crypto_held = 0

                # Record the trade
                self.trades_done.append(
                    {
                        "step": self.current_step,
                        "type": "sell",
                        "price": current_price,
                        "amount": sell_amount,
                        "proceeds": sell_amount * current_price * (1 - self.commission),
                    }
                )

        # Advance to next step
        self.current_step += 1

        # Update portfolio value
        self.portfolio_value = self.balance + self.crypto_held * self.current_price
        self.portfolio_values.append(self.portfolio_value)

        # Calculate reward based on portfolio returns
        reward = (
            (self.portfolio_value / prev_portfolio_value) - 1
        ) * self.reward_scaling

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        # Get new observation
        observation = self._get_observation()

        # Return step information
        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "crypto_held": self.crypto_held,
            "current_price": current_price,
            "step": self.current_step,
        }

        return observation, reward, done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment.

        For now, this just prints the current state.
        """
        print(f"Step: {self.current_step}")
        print(f"Price: {self.current_price:.2f}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Crypto Held: {self.crypto_held:.8f}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print("----------------------------")

        return None
