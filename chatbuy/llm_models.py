import os
# Load environment variables from a .env file
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
# Import the OpenAIModel class from pydantic_ai
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables, overriding existing ones if necessary
load_dotenv(override=True)

# Initialize an AsyncAzureOpenAI client for the 0806 model
client_0806 = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_0806"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION_0806"],
    api_key=os.environ["AZURE_OPENAI_API_KEY_0806"],
)

# Create an OpenAIModel instance for the 0806 model
model_0806 = OpenAIModel("4o0806", openai_client=client_0806)

# Initialize another AsyncAzureOpenAI client for the default model
client = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

# Create OpenAIModel instances for the gpt4o and gpt4o-mini models
model_4o = OpenAIModel("gpt4o", openai_client=client)
model_4o_mini = OpenAIModel("gpt4o-mini", openai_client=client)

if __name__ == "__main__":
    # Import the Agent class from pydantic_ai
    from pydantic_ai import Agent

    # Create an Agent instance with the gpt4o-mini model and a system prompt
    agent = Agent(
        model=model_4o_mini,
        system_prompt="Be concise, reply with one sentence.",
    )

    # Run the agent synchronously with a sample query and print the result
    result = agent.run_sync('Where does "hello world" come from?')
    print(result.data)
