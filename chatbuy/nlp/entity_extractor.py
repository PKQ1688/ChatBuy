import re
from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    """Extracted entity from text."""
    entity_type: str
    value: int | float | str
    start_pos: int
    end_pos: int


class EntityExtractor:
    """Extract trading parameters from natural language."""
    
    def __init__(self):
        self.patterns = {
            "fast_period": [
                r"快线.*?(\d+)",
                r"短期.*?(\d+)",
                r"fast.*?(\d+)",
                r"短周期.*?(\d+)",
                r"(\d+).*?日.*?快",
            ],
            "slow_period": [
                r"慢线.*?(\d+)",
                r"长期.*?(\d+)",
                r"slow.*?(\d+)",
                r"长周期.*?(\d+)",
                r"(\d+).*?日.*?慢",
            ],
            "symbol": [
                r"BTC|bitcoin|比特币",
                r"ETH|ethereum|以太坊",
                r"AAPL|apple|苹果",
                r"TSLA|tesla|特斯拉",
            ],
            "rsi_period": [
                r"rsi.*?(\d+)",
                r"相对强弱.*?(\d+)",
            ],
            "rsi_lower": [
                r"rsi.*?低于.*?(\d+)",
                r"rsi.*?小于.*?(\d+)",
                r"超卖.*?(\d+)",
            ],
            "rsi_upper": [
                r"rsi.*?高于.*?(\d+)",
                r"rsi.*?大于.*?(\d+)",
                r"超买.*?(\d+)",
            ],
        }
    
    def extract(self, text: str) -> dict[str, int | float | str]:
        """Extract all entities from text."""
        entities = {}
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    # Convert to appropriate type
                    if entity_type in ["fast_period", "slow_period", "rsi_period"]:
                        entities[entity_type] = int(value)
                    elif entity_type in ["rsi_lower", "rsi_upper"]:
                        entities[entity_type] = int(value)
                    elif entity_type == "symbol":
                        entities[entity_type] = self._normalize_symbol(match.group(0))
                    break
        
        return entities
    
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
        """Specifically extract moving average parameters."""
        entities = self.extract(text)
        
        # Set default values if not found
        if "fast_period" not in entities:
            entities["fast_period"] = 20
        if "slow_period" not in entities:
            entities["slow_period"] = 50
        
        # Ensure the values are integers
        fast_period = int(entities["fast_period"]) if "fast_period" in entities else 20
        slow_period = int(entities["slow_period"]) if "slow_period" in entities else 50
        
        return {
            "fast_period": fast_period,
            "slow_period": slow_period
        }