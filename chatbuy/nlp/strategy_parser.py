
from typing import Any

from .entity_extractor import EntityExtractor
from .intent_classifier import IntentClassifier


class StrategyParser:
    """Main parser for converting natural language to strategy parameters."""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
    
    def parse(self, text: str) -> dict[str, Any] | None:
        """Parse natural language description into strategy parameters."""
        # Classify intent
        intent = self.intent_classifier.classify(text)
        
        if intent.strategy_type == "unknown":
            return None
        
        # Extract entities based on strategy type
        if intent.strategy_type == "moving_average_cross":
            parameters = self.entity_extractor.extract_ma_parameters(text)
            strategy_config = {
                "strategy_type": "moving_average_cross",
                "parameters": parameters,
                "confidence": intent.confidence
            }
        else:
            # For other strategy types, extract general entities
            parameters = self.entity_extractor.extract(text)
            strategy_config = {
                "strategy_type": intent.strategy_type,
                "parameters": parameters,
                "confidence": intent.confidence
            }
        
        return strategy_config
    
    def parse_ma_strategy(self, text: str) -> dict[str, int] | None:
        """Specifically parse moving average strategy parameters."""
        try:
            return self.entity_extractor.extract_ma_parameters(text)
        except Exception:
            return None
    
    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate extracted parameters."""
        if "fast_period" in parameters and "slow_period" in parameters:
            return parameters["fast_period"] < parameters["slow_period"]
        return True