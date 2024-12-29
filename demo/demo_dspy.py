import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
from rich.pretty import pprint

from chatbuy.base_model.dspy_lm import lm_4o_mini as lm

dspy.configure(lm=lm)


def search(query: str) -> list[str]:
    """Retrieves abstracts from Wikipedia."""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(
        query, k=3
    )
    return [x['text'] for x in results]


trainset = [
    x.with_inputs('question')
    for x in HotPotQA(train_seed=2024, train_size=50, dev_size=500).train
]

devset = [
    x.with_inputs('question')
    for x in HotPotQA(train_seed=2023, train_size=50, dev_size=500).dev
]

# breakpoint()

pprint(trainset)
react = dspy.ReAct('question -> answer', tools=[search])
# react.load("demo/optimized_react.json")


def validate_answer(example, pred, trace=None):
    """Validates if the predicted answer matches the example answer, case-insensitive.

    Args:
        example: The example object containing the correct answer.
        pred: The prediction object containing the predicted answer.
        trace: Optional trace information (default: None).

    Returns:
        bool: True if the predicted answer matches the example answer (case-insensitive), False otherwise.
    """
    return example.answer.lower() == pred.answer.lower()


evaluator = Evaluate(
    devset=devset, num_threads=1, display_progress=True, display_table=5
)
evaluator(react, metric=validate_answer)

# tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
# optimized_react = tp.compile(react, trainset=trainset)

# optimized_react.save("demo/optimized_react.json")
# react.load("demo/optimized_react.json")
# pred = react(question=trainset[0].question)

# print(pred)
#
# lm.inspect_history()
