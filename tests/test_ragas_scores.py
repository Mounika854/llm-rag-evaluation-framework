from src.evaluator import Evaluator

def test_ragas_scores():
    evaluator = Evaluator()

    question = "Where is Hyderabad?"
    answer = "Hyderabad is in Telangana, India."
    contexts = ["Hyderabad is a major city located in Telangana, India."]

    results = evaluator.evaluate_rag(question, answer, contexts)

    # Ensure score object returns numeric values
    for metric, score in results.items():
        assert score >= 0
        assert score <= 1
