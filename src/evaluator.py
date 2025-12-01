from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
)

class Evaluator:

    def evaluate_rag(self, question, answer, contexts):
        """Run RAGAS evaluation with multiple metrics."""
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }

        score = evaluate(
            data,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
        )

        return score
