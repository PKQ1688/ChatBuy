import re
from dataclasses import dataclass
from typing import Any


@dataclass
class StrategyIntent:
    """Strategy intent extracted from text."""
    strategy_type: str
    parameters: dict[str, Any]
    confidence: float


class IntentClassifier:
    """Classify user intent from natural language description."""
    
    def __init__(self):
        self.strategy_patterns = {
            "moving_average_cross": [
                r"双均线.*?金叉",
                r"均线.*?交叉",
                r"ma.*?cross",
                r"moving.*?average.*?cross",
                r"快线.*?慢线",
                r"短期.*?长期.*?均线",
            ],
            "bollinger_bands": [
                r"布林带",
                r"bollinger.*?band",
                r"布林轨",
            ],
            "rsi_oversold": [
                r"rsi.*?超卖",
                r"rsi.*?oversold",
                r"rsi.*?低于.*?[0-9]+",
            ],
            "rsi_overbought": [
                r"rsi.*?超买",
                r"rsi.*?overbought",
                r"rsi.*?高于.*?[0-9]+",
            ],
        }
    
    def classify(self, text: str) -> StrategyIntent:
        """Classify the user's intent from text description."""
        text = text.lower()
        best_match = None
        highest_confidence = 0.0
        
        for strategy_type, patterns in self.strategy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    confidence = self._calculate_confidence(text, pattern)
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = strategy_type
        
        if best_match:
            return StrategyIntent(
                strategy_type=best_match,
                parameters={},
                confidence=highest_confidence
            )
        
        return StrategyIntent(
            strategy_type="unknown",
            parameters={},
            confidence=0.0
        )
    
    def _calculate_confidence(self, text: str, pattern: str) -> float:
        """Calculate confidence score for pattern match."""
        # Simple confidence based on pattern specificity
        if "金叉" in pattern or "cross" in pattern:
            return 0.9
        elif "均线" in pattern or "ma" in pattern:
            return 0.8
        elif "rsi" in pattern:
            return 0.7
        else:
            return 0.6