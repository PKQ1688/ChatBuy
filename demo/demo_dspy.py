import dspy
from scripts.dspy_lm import lm_4o_mini as lm
from dspy.datasets import HotPotQA
from rich.pretty import pprint

dspy.configure(lm=lm)


def search(query: str) -> list[str]:
    """Retrieves abstracts from Wikipedia."""
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(
        query, k=3
    )
    return [x["text"] for x in results]


trainset = [
    x.with_inputs("question") for x in HotPotQA(train_seed=2024, train_size=500).train
]

pprint(trainset)

# react = dspy.ReAct("question -> answer", tools=[search])

# tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
# optimized_react = tp.compile(react, trainset=trainset)

# optimized_react.save("demo/optimized_react.json")
# react.load("demo/optimized_react.json")
# pred = react(question=trainset[0].question)

# print(pred)
# 
# lm.inspect_history()
