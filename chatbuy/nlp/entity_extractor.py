import json
import os
from dataclasses import dataclass

import openai
from dotenv import load_dotenv


@dataclass
class ExtractedEntity:
    """Extracted entity from text."""

    entity_type: str
    value: int | float | str
    start_pos: int
    end_pos: int


class EntityExtractor:
    """Extract trading parameters from natural language using LLM."""

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

    def extract(self, text: str) -> dict[str, int | float | str]:
        """Extract all entities from text using LLM."""
        prompt = f"""
        You are a trading strategy parameter extraction expert. Extract all relevant trading parameters from the following description.
        
        User description: "{text}"
        
        Please respond with a JSON object containing the extracted parameters. Common parameters include:
        - fast_period: Integer for fast moving average period
        - slow_period: Integer for slow moving average period
        - symbol: String for trading symbol (e.g., "BTC-USD", "ETH-USD", "AAPL")
        - rsi_period: Integer for RSI period
        - rsi_lower: Integer for RSI oversold threshold
        - rsi_upper: Integer for RSI overbought threshold
        
        Examples:
        - "双均线金叉买入，20日均线和50日均线" -> {{"fast_period": 20, "slow_period": 50}}
        - "快线10日，慢线30日，金叉买入死叉卖出" -> {{"fast_period": 10, "slow_period": 30}}
        - "RSI低于30买入" -> {{"rsi_lower": 30}}
        - "BTC的双均线策略" -> {{"symbol": "BTC-USD", "fast_period": 20, "slow_period": 50}}
        
        If a parameter is not mentioned, don't include it in the JSON. Use default values in the calling code.
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

        # Normalize symbol if present
        if "symbol" in result:
            result["symbol"] = self._normalize_symbol(result["symbol"])

        return result

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol names."""
        symbol_map = {
            "BTC": "BTC-USD",
            "bitcoin": "BTC-USD",
            "比特币": "BTC-USD",
            "ETH": "ETH-USD",
            "ethereum": "ETH-USD",
            "以太坊": "ETH-USD",
            "AAPL": "AAPL",
            "apple": "AAPL",
            "苹果": "AAPL",
            "TSLA": "TSLA",
            "tesla": "TSLA",
            "特斯拉": "TSLA",
        }
        return symbol_map.get(symbol.lower(), symbol.upper())

    def extract_ma_parameters(self, text: str) -> dict[str, int]:
        """Specifically extract moving average parameters using LLM."""
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
