from src.rag_pipeline import RAGPipeline

def test_context_precision():
    rag = RAGPipeline()

    text = "Python is a programming language. Mango is a fruit."
    chunks = rag.split_text(text)
    db = rag.create_vector_db(chunks)

    query = "What is Python?"
    retrieved = rag.retrieve(db, query)

    assert len(retrieved) > 0
    assert "programming language" in retrieved[0].page_content
