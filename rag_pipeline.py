
class RAGPipeline:
    """Simple RAG pipeline."""

    def __init__(self):
        self.documents = []

    def load_documents(self):
        print("Loading documents...")

    def create_embeddings(self):
        print("Generating vector embeddings...")

    def answer_question(self, question):
        return f"Answer generated for: {question}"
