import json
import os
from typing import Any

import openai
from dotenv import load_dotenv


class StrategyParser:
    """Main parser for converting natural language to strategy parameters using LLM."""

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

    def parse(self, text: str) -> dict[str, Any] | None:
        """Parse natural language description into strategy parameters using LLM."""
        prompt = f"""
        You are a trading strategy expert. Analyze the following trading strategy description and extract all relevant information.
        
        User description: "{text}"
        
        Please respond with a JSON object containing:
        - strategy_type: One of ["moving_average_cross", "bollinger_bands", "rsi_oversold", "rsi_overbought", "unknown"]
        - parameters: Extracted parameters (fast_period, slow_period, symbol, rsi_period, rsi_lower, rsi_upper, etc.)
        - confidence: Float between 0.0 and 1.0
        
        Examples:
        - "双均线金叉买入，20日均线和50日均线" -> {{"strategy_type": "moving_average_cross", "parameters": {{"fast_period": 20, "slow_period": 50}}, "confidence": 0.95}}
        - "快线10日，慢线30日，金叉买入死叉卖出" -> {{"strategy_type": "moving_average_cross", "parameters": {{"fast_period": 10, "slow_period": 30}}, "confidence": 0.9}}
        - "RSI低于30买入" -> {{"strategy_type": "rsi_oversold", "parameters": {{"rsi_lower": 30}}, "confidence": 0.9}}
        - "布林带策略" -> {{"strategy_type": "bollinger_bands", "parameters": {{}}, "confidence": 0.85}}
        
        If parameters are not specified, use reasonable defaults:
        - moving_average_cross: fast_period=20, slow_period=50
        - rsi_oversold: rsi_lower=30
        - rsi_overbought: rsi_upper=70
        
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

        if result.get("strategy_type") == "unknown":
            return None

        # Validate and normalize parameters
        parameters = result.get("parameters", {})
        parameters = self._normalize_parameters(result.get("strategy_type"), parameters)

        return {
            "strategy_type": result.get("strategy_type"),
            "parameters": parameters,
            "confidence": float(result.get("confidence", 0.0)),
        }

    def _normalize_parameters(
        self, strategy_type: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize and validate parameters based on strategy type."""
        if strategy_type == "moving_average_cross":
            # Ensure fast_period < slow_period
            fast_period = parameters.get("fast_period", 20)
            slow_period = parameters.get("slow_period", 50)

            if fast_period >= slow_period:
                # Swap if needed
                fast_period, slow_period = slow_period, fast_period

            return {"fast_period": int(fast_period), "slow_period": int(slow_period)}

        elif strategy_type == "rsi_oversold":
            return {"rsi_lower": int(parameters.get("rsi_lower", 30))}

        elif strategy_type == "rsi_overbought":
            return {"rsi_upper": int(parameters.get("rsi_upper", 70))}

        elif strategy_type == "bollinger_bands":
            return parameters  # No specific normalization needed

        return parameters

    def parse_ma_strategy(self, text: str) -> dict[str, int] | None:
        """Specifically parse moving average strategy parameters using LLM."""
        prompt = f"""
        You are a trading strategy parameter extraction expert. Extract moving average parameters from the following description.
        
        User description: "{text}"
        
        Please respond with a JSON object containing:
        - fast_period: Integer for fast moving average period (default: 20 if not specified)
        - slow_period: Integer for slow moving average period (default: 50 if not specified)
        
        Examples:
        - "双均线金叉买入，20日均线和50日均线" -> {{"fast_period": 20, "slow_period": 50}}
        - "快线10日，慢线30日" -> {{"fast_period": 10, "slow_period": 30}}
        - "20和60日均线交叉" -> {{"fast_period": 20, "slow_period": 60}}
        - "双均线策略" -> {{"fast_period": 20, "slow_period": 50}}
        
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

        # Set default values if not found
        fast_period = result.get("fast_period", 20)
        slow_period = result.get("slow_period", 50)

        return {"fast_period": int(fast_period), "slow_period": int(slow_period)}

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate extracted parameters."""
        if "fast_period" in parameters and "slow_period" in parameters:
            return parameters["fast_period"] < parameters["slow_period"]
        return True
