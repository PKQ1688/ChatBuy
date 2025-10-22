import json
import os
from dataclasses import dataclass

import openai
from dotenv import load_dotenv

# Strategy parameter value types
type StrategyParamValue = (
    int | float | str | bool | list[dict[str, str | int | float]] | dict[str, list[int]]
)


@dataclass
class StrategyIntent:
    """Strategy intent extracted from text."""

    strategy_type: str
    parameters: dict[str, StrategyParamValue]
    confidence: float


class IntentClassifier:
    """Classify user intent from natural language description using LLM."""

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize OpenAI client (env vars may be absent during analysis)
        api_key = os.getenv("API_KEY") or None
        base_url = os.getenv("MODEL_URL") or None
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))

    def classify(self, text: str) -> StrategyIntent:
        """Classify the user's intent from text description using LLM."""
        prompt = f"""
        You are a trading strategy expert. Analyze the following trading strategy description and classify it.

        User description: "{text}"

        Please respond with a JSON object containing:
        - strategy_type: One of ["moving_average_cross", "bollinger_bands", "rsi_oversold", "rsi_overbought", "unknown"]
        - confidence: A float between 0.0 and 1.0 indicating your confidence in the classification
        - parameters: Any relevant parameters you can extract from the description

        Examples:
        - "dual moving average golden cross buy, 20-day and 50-day moving average" -> {{"strategy_type": "moving_average_cross", "confidence": 0.95, "parameters": {{"fast_period": 20, "slow_period": 50}}}}
        - "RSI below 30 buy" -> {{"strategy_type": "rsi_oversold", "confidence": 0.9, "parameters": {{"rsi_lower": 30}}}}
        - "Bollinger bands strategy" -> {{"strategy_type": "bollinger_bands", "confidence": 0.85, "parameters": {{}}}}

        Respond with only the JSON object, no other text.
        """

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        response_text = response.output_text.strip()
        result = json.loads(response_text)

        return StrategyIntent(
            strategy_type=result.get("strategy_type", "unknown"),
            parameters=result.get("parameters", {}),
            confidence=float(result.get("confidence", 0.0)),
        )
