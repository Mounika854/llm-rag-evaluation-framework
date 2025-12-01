from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pypdf import PdfReader


class RAGPipeline:

    def load_pdf(self, file_path):
        """Load PDF and return its text."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def split_text(self, text):
        """Split text into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return splitter.split_text(text)

    def create_vector_db(self, chunks):
        """Create FAISS vector database."""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docs = [Document(page_content=chunk) for chunk in chunks]
        db = FAISS.from_documents(docs, embeddings)
        return db

    def retrieve(self, db, query, k=3):
        """Retrieve top documents."""
        return db.similarity_search(query, k=k)
