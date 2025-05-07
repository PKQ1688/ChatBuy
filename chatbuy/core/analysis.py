# from textwrap import dedent

import glob
import os

from agno.agent import Agent
from agno.media import Image
from agno.models.azure import AzureOpenAI


class BasicAnalysis:
    """Represents basic analysis of a chart."""

    def __init__(
        self,
        model_name: str = "gpt-4o-1120",
        temperature: float = 0.1,
        system_prompt: str = None,
    ):
        if system_prompt is None:
            system_prompt = "You are a trading assistant. You can analyze K-line charts and provide trading advice."

        self.model = AzureOpenAI(id=model_name, temperature=temperature)

        self.agent = Agent(
            model=self.model,
            description=system_prompt,
            instructions=[],
        )

    def analyze_images(
        self,
        image_dir="data/btc_daily_refactored",
        pattern="*.png",
        prompt="请分析这张K线图，并给出你的交易建议。",
    ):
        """Batch analyze all images in the specified folder and return a list of image paths and AI analysis results."""
        image_paths = glob.glob(os.path.join(image_dir, pattern))
        results = []
        for img_path in sorted(image_paths):
            images = [Image(filepath=img_path)]
            try:
                response = self.agent.run(prompt, images=images)
                results.append({"image": img_path, "result": response})
            except Exception as e:
                results.append({"image": img_path, "result": f"分析失败: {e}"})
        return results
