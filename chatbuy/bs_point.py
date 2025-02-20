# from enum import Enum

# import logfire
# from pydantic import BaseModel
# from pydantic_ai import Agent, Tool

# from chatbuy.base_model.pydantic_lm import model_1120 as llm_model
# from chatbuy.tool.technicals import fake_technical_analyst

# logfire.configure()


# class StrategyEnum(str, Enum):
#     """Enumeration of possible trading strategies."""

#     LONG = "LONG"
#     SHORT = "SHORT"
#     HOLD = "HOLD"


# class TradingDecision(BaseModel):
#     """A model representing a trading decision."""

#     Strategy: StrategyEnum
#     Reason: str
#     Confidence: float


# buy_agent = Agent(
#     model=llm_model,
#     # system_prompt='Make a trading decision based on the provided data.',
#     tools=[Tool(fake_technical_analyst, takes_ctx=False)],
#     result_type=TradingDecision,
# )


# res = buy_agent.run_sync("Make a trading decision based on the provided data.")
# print(res.data.Strategy.value)
# print(res.data.Reason)
# print(res.data.Confidence)
