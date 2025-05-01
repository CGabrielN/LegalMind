"""
LegalMind Basic RAG Implementation

This module implements the basic RAG (Retrieval Augmented Generation) pipeline
for the LegalMind system.
"""

import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..vectordb.chroma_db import ChromaVectorStore
from ..embeddings.embedding import EmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class BasicRAG:
    """
    Implements basic Retrieval Augmented Generation for legal queries.
    """

    def __init__(self):
        """Initialize the basic RAG system."""
        self.vector_store = ChromaVectorStore()
        self.embedding_model = EmbeddingModel()
        self.top_k = config["rag"]["basic"]["top_k"]

        logger.info("Initialized Basic RAG system")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant legal documents for a query.

        Args:
            query: The user's legal query
            top_k: Number of documents to retrieve (overrides config)

        Returns:
            List of relevant documents with text and metadata
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"Retrieving top {top_k} documents for query: '{query[:50]}...'")

        # Embed the query
        query_embedding = self.embedding_model.embed_text(query)

        # Query the vector store
        results = self.vector_store.query(query, n_results=top_k)

        # Format the results
        documents = []
        for i in range(len(results["documents"][0])):
            document = {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                "id": results["ids"][0][i] if results["ids"][0] else f"doc_{i}",
                "score": 1.0 - results["distances"][0][i] if results["distances"][0] else 0.0
            }
            documents.append(document)

        return documents

    def prepare_context(self, documents: List[Dict[str, Any]], max_tokens: int = 3800) -> str:
        """
        Prepare retrieved documents as context for the LLM.

        Args:
            documents: List of retrieved documents
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        logger.info(f"Preparing context from {len(documents)} documents")

        context_parts = []
        current_tokens = 0

        for i, doc in enumerate(documents):
            # Estimate tokens (rough approximation)
            doc_tokens = len(doc["text"].split())

            # If adding this document would exceed max tokens, skip it
            if current_tokens + doc_tokens > max_tokens:
                logger.info(f"Reached token limit with {i} documents")
                break

            # Format document with metadata
            doc_context = f"Document {i+1}:\n"

            # Add any citation information
            if "citation" in doc["metadata"]:
                doc_context += f"Citation: {doc['metadata']['citation']}\n"

            # Add jurisdiction if available
            if "jurisdiction" in doc["metadata"]:
                doc_context += f"Jurisdiction: {doc['metadata']['jurisdiction']}\n"

            # Add document type if available
            if "document_type" in doc["metadata"]:
                doc_context += f"Document Type: {doc['metadata']['document_type']}\n"

            # Add the document text
            doc_context += f"Content:\n{doc['text']}\n\n"

            context_parts.append(doc_context)
            current_tokens += doc_tokens

        # Join all context parts
        full_context = "\n".join(context_parts)

        return full_context

    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a query through the retrieval pipeline.

        Args:
            query: The user's legal query

        Returns:
            Tuple of (context_for_llm, retrieved_documents)
        """
        logger.info(f"Processing query: '{query[:50]}...'")

        # Retrieve relevant documents
        documents = self.retrieve(query)

        # Prepare context for LLM
        context = self.prepare_context(documents)

        return context, documents