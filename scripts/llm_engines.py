import os

from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI
from pydantic import BaseModel
from transformers.agents.llm_engine import MessageRole, get_clean_message_list

load_dotenv(override=True)

openai_role_conversions = {MessageRole.TOOL_RESPONSE: MessageRole.USER}


class OpenAIEngine:
    def __init__(self, model_name="gpt4o"):
        self.model_name = model_name
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    def __call__(self, messages, stop_sequences=None, grammar=None):
        if stop_sequences is None:
            stop_sequences = []
        messages = get_clean_message_list(
            messages,
            role_conversions=openai_role_conversions,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.7,
            response_format=grammar,
        )
        return response.choices[0].message.content


class OpenAIEngineStructuredOutputs:
    def __init__(self, model_name="4o0806"):
        self.model_name = model_name
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY_0806"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_0806"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION_0806"),
        )

    def __call__(self, messages, stop_sequences=None, grammar=None):
        if stop_sequences is None:
            stop_sequences = []

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.7,
            response_format=grammar,
        )
        return completion.choices[0].message.parsed


llm_engine_4o = OpenAIEngine(model_name="gpt4o")
llm_engine_4o_mini = OpenAIEngine(model_name="gpt4o-mini")
llm_engine_0806 = OpenAIEngineStructuredOutputs(model_name="4o0806")


if __name__ == "__main__":

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    messages = [
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday",
        },
    ]
    response1 = llm_engine_4o(messages)
    response2 = llm_engine_4o_mini(messages)
    response3 = llm_engine_0806(messages, grammar=CalendarEvent)
    
    print("=" * 30)
    print(response1)
    print('-' * 30)
    print(response2)
    print('-' * 30)
    print(response3)
    print("=" * 30)
