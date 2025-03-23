# import os

from base_model.smol_lm import model_1120 as model
from mcp import StdioServerParameters
from smolagents import CodeAgent, ToolCallingAgent, ToolCollection

# server_parameters = StdioServerParameters(
#     command="uv",
#     args=["--quiet", "pubmedmcp@0.1.3"],
#     env={"UV_PYTHON": "3.12", **os.environ},
# )
use_code = False

server_parameters = StdioServerParameters(
    command="uvx",  # Executable
    args=["mcp-server-fetch"],  # Optional command line arguments
    # env=None # Optional environment variables
)


# server_parameters = StdioServerParameters(
#     command="docker",  # Executable
#     args=[
#         "run",
#         "-i",
#         "--rm",
#         "--init",
#         "-e",
#         "DOCKER_CONTAINER=true",
#         "mcp/puppeteer",
#     ],  # Optional command line arguments
#     # env=None # Optional environment variables
# )


with ToolCollection.from_mcp(server_parameters) as tool_collection:

    if use_code:
        agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=False, model=model)
    else:
        agent = ToolCallingAgent(
            tools=[*tool_collection.tools], model=model, add_base_tools=False
        )
    agent.run(
        "帮我使用中文总结一下这个 https://huggingface.co/blog/smolagents-phoenix 网页的内容"
    )
