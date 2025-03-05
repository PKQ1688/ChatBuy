from smolagents import HfApiModel, tool
from smolagents.agents import CodeAgent, ToolCallingAgent

# Choose which inference type to use!


model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct")


@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


agent = ToolCallingAgent(tools=[get_weather], model=model)

print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))

agent = CodeAgent(tools=[get_weather], model=model)

print("CodeAgent:", agent.run("What's the weather like in Paris?"))
