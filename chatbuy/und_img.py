import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv(override=True)

provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)
model = OpenAIModel("openrouter/quasar-alpha", provider=provider)

image_path = "data/BTC_indicators_20250407_211726.png"  # 替换为你的本地图片路径
with open(image_path, "rb") as image_file:
    image_content = image_file.read()


class TradeAdvice(BaseModel):
    """Represents trade advice based on the analysis of a chart.

    action : str
        The recommended action, which can be "hold", "sell", or "buy".
    reason : str
        The reason for the recommended action.
    """
    action: str  # Possible values: "hold", "sell", "buy"
    reason: str


agent = Agent(model=model, result_type=TradeAdvice)
result = agent.run_sync(
    [
        "帮我分析一下这个图表的走势",
        BinaryContent(data=image_content, media_type="image/png"),
    ]
)
print(result.data)
