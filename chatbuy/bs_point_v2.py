from typing import Literal

from pydantic import BaseModel
from smolagents import tool
from smolagents.agents import ToolCallingAgent

from chatbuy.base_model.smol_lm import model_1120 as model
from chatbuy.tool.technicals import fake_technical_analyst


@tool
def get_technical_analysis() -> str:
    """Get the technical analysis of the provided data."""
    return fake_technical_analyst()


class TradingDecision(BaseModel):
    """A model representing a trading decision."""

    Strategy: str = Literal["LONG", "SHORT", "HOLD"]
    Reason: str


agent = ToolCallingAgent(
    tools=[get_technical_analysis], model=model, grammar=TradingDecision
)

print(agent.run("Make a trading decision based on the provided data."))
