from src.rag_pipeline import RAGPipeline

def test_context_recall():
    rag = RAGPipeline()

    text = "India has 28 states. Hyderabad is the capital of Telangana."
    chunks = rag.split_text(text)
    db = rag.create_vector_db(chunks)

    query = "How many states does India have?"
    retrieved = rag.retrieve(db, query)

    assert len(retrieved) > 0
    assert "28 states" in retrieved[0].page_content
