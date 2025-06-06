import os
import logging
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

from utilis import config  # Loads the validated config object

class RetrieverAgent:
    def __init__(self, collection_name="medical_docs"):
        """Initialize ChromaDB retriever agent."""
        config.validate_config()
        config.setup_logging()

        self.logger = logging.getLogger("RetrieverAgent")

        self.logger.info("Initializing RetrieverAgent...")
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )

        self.client = chromadb.PersistentClient(path=config.VECTOR_STORE_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

        self.logger.info(f"ChromaDB initialized at: {config.VECTOR_STORE_PATH}")

    def add_documents(self, texts: list[str]):
        """Add new documents to the collection."""
        self.logger.info(f"Adding {len(texts)} documents...")
        existing_ids = self.collection.get()['ids']
        start_idx = len(existing_ids)

        for i, doc in enumerate(texts):
            doc_id = f"doc-{start_idx + i}"
            self.collection.add(documents=[doc], ids=[doc_id])

        self.logger.info("Documents added and indexed.")

    def retrieve(self, query: str, top_k: int = None) -> str:
        """Retrieve top-k relevant documents for a given query."""
        k = top_k or config.MAX_RETRIEVED_DOCS
        self.logger.debug(f"Query: '{query}' | Top-k: {k}")

        results = self.collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        return "\n---\n".join(docs) if docs else "No relevant documents found."


# Optional: For quick testing
if __name__ == "__main__":
    agent = RetrieverAgent()

    # Add sample docs (you can skip if already persisted)
    agent.add_documents([
        "Chest X-rays help identify lung infections and heart enlargement.",
        "MRI scans are ideal for evaluating soft tissues like brain and muscles.",
        "CT is useful for viewing complex fractures and organ damage."
    ])

    query = "How can MRI be used in soft tissue diagnosis?"
    print(agent.retrieve(query))
