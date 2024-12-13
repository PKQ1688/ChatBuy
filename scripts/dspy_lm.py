import os

import dspy
from dotenv import load_dotenv

load_dotenv()


lm_4o = dspy.LM(
    model="azure/gpt4o",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

lm_4o_mini = dspy.LM(
    model="azure/gpt4o-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

lm_4o_0806 = dspy.LM(
    model="azure/4o0806",
    api_key=os.getenv("AZURE_OPENAI_API_KEY_0806"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_0806"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT_0806"),
)

lm_o1_mini = dspy.LM(
    model="azure/o1-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=1.0,
    max_tokens=10000,
)

# lm_o1_preview = dspy.LM(
#     model="azure/o1-preivew",
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     temperature=1.0,
#     max_completion_tokens=10000,
# )

if __name__ == "__main__":
    res = lm_4o("who are you", temperature=0.7)
    print("4o:", res[0])

    res = lm_4o_mini("who are you", temperature=0.7)
    print("4o mini:", res[0])

    res = lm_4o_0806("who are you", temperature=0.7)
    print("4o 0806:", res[0])

    res = lm_o1_mini("who are you")
    print("o1 mini:", res[0])

    # res = lm_o1_preview("who are you")
    # print("o1 preview:", res[0])
