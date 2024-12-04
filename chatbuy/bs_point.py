from phi.agent import Agent
from phi.tools.openbb_tools import OpenBBTools
from phi.model.azure import AzureOpenAIChat


agent = Agent(
    model=AzureOpenAIChat(id="gpt4o"),
    tools=[OpenBBTools(provider="polygon")],
    debug_mode=True,
    show_tool_calls=True,
)

# Example usage showing stock analysis
agent.print_response(
    "今天BTCUSDT的价格是好多"
)

# # Example showing market analysis
# agent.print_response("What are the top gainers in the market today?")

# # Example showing economic indicators
# agent.print_response(
#     "Show me the latest GDP growth rate and inflation numbers for the US"
# )
