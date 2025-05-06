# from textwrap import dedent

from agno.agent import Agent

# from agno.media import Image
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

    def gen_image(self):
        """Generates an image based on the analysis."""
        pass