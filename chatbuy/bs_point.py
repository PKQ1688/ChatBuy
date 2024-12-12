from pathlib import Path

from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from scripts.llm_models import model_4o_mini as llm_model

cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

python_agent = PythonAgent(
    model=llm_model,
    base_dir=tmp,
    files=[
        CsvFile(
            path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
            description="Contains information about movies from IMDB.",
        )
    ],
    markdown=True,
    pip_install=True,
    show_tool_calls=True,
)
python_agent.print_response("What is the average rating of movies?")
