# from textwrap import dedent

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

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Please analyze this K-line chart and provide your trading advice.",
    ):
        """Analyzes a single image and returns its path and AI analysis result."""
        if not os.path.exists(image_path):
            return {"image": image_path, "result": f"Image not found at {image_path}."}

        if not os.path.isfile(image_path):
            return {"image": image_path, "result": f"Path {image_path} is not a file."}

        images = [Image(filepath=image_path)]
        try:
            response = self.agent.run(prompt, images=images)
            return {"image": image_path, "result": response.content}
        except Exception as e:
            return {"image": image_path, "result": f"Analysis failed: {e}"}


if __name__ == "__main__":
    # Example usage
    analysis = BasicAnalysis()
    result = analysis.analyze_images(
        "data/btc_daily_refactored/btc_daily_20180116_len120.png"
    )
    print(result)
