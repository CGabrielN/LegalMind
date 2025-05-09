"""
LegalMind Multi-Query RAG Strategy

This module implements a multi-query approach to capture different legal perspectives
in the same question by generating multiple query versions and combining results.
"""

import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiQueryRAG:
    """
    Implements a multi-query RAG strategy to capture different legal perspectives.
    """

    def __init__(self, resource_manager=None):
        """
        Initialize the multi-query RAG system.

        Args:
            resource_manager: Shared ResourceManager instance
        """

        # Import here to avoid circular imports
        if resource_manager is None:
            from src.core.resource_manager import ResourceManager
            resource_manager = ResourceManager()

        self.resource_manager = resource_manager

        # Access shared resources through the manager
        self.vector_store = resource_manager.vector_store
        self.embedding_model = resource_manager.embedding_model
        self.query_expander = resource_manager.query_expander

        # Load configuration
        self.config = resource_manager.config
        self.top_k = self.config["rag"]["multi_query"]["top_k"]

        logger.info("Initialized Multi-Query RAG system")

    def generate_query_perspectives(self, query: str) -> List[str]:
        """
        Generate different perspective queries for the same legal question.

        Args:
            query: The user's legal query

        Returns:
            List of query variations
        """
        # Start with the original query
        perspectives = [query]

        # Generate jurisdiction-specific perspectives
        jurisdiction = self.query_expander.identify_jurisdiction(query)
        legal_terms = self.query_expander.identify_legal_terms(query)

        # If we identified legal terms but no jurisdiction, generate perspectives
        # for key Australian jurisdictions
        if legal_terms and not jurisdiction:
            for jur in ["nsw", "commonwealth"]:
                jur_terms = self.query_expander.jurisdiction_terminology.get(jur, {})
                for category, terms in jur_terms.items():
                    if terms:
                        perspectives.append(f"{query} {terms[0]}")

        # Add plaintiff perspective
        if any(term in query.lower() for term in ["negligence", "contract", "damages", "injury"]):
            perspectives.append(f"{query} from plaintiff perspective rights")

        # Add defendant perspective
        if any(term in query.lower() for term in ["negligence", "contract", "damages", "liability"]):
            perspectives.append(f"{query} from defendant perspective defense")

        # Add statutory perspective
        if any(term in query.lower() for term in ["law", "legal", "right", "obligation"]):
            perspectives.append(f"{query} statutory provisions australia")

        # Add case law perspective
        perspectives.append(f"{query} relevant case law precedent")

        # Make sure we have at least 3 perspectives
        if len(perspectives) < 3:
            # Add a broad legal principles perspective
            perspectives.append(f"{query} legal principles australia")

        logger.info(f"Generated query perspectives: {perspectives}")
        return perspectives

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multiple query perspectives.

        Args:
            query: The user's legal query

        Returns:
            List of relevant documents with text and metadata
        """
        logger.info(f"Multi-query retrieval for: '{query[:50]}...'")

        # Generate query perspectives
        perspectives = self.generate_query_perspectives(query)

        # Results storage
        all_results = []
        doc_ids = set()  # Track seen document IDs to avoid duplicates

        # Retrieve documents for each perspective
        for perspective in perspectives:
            # Embed the perspective query
            query_embedding = self.embedding_model.embed_text(perspective)

            # Query the vector store
            results = self.vector_store.query(perspective, n_results=self.top_k)

            # Process results
            for i in range(len(results["documents"][0])):
                doc_id = results["ids"][0][i] if results["ids"][0] else f"doc_{i}"

                # Skip if we've already seen this document
                if doc_id in doc_ids:
                    continue

                doc_ids.add(doc_id)

                document = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    "id": doc_id,
                    "score": 1.0 - results["distances"][0][i] if results["distances"][0] else 0.0,
                    "perspective": perspective
                }

                all_results.append(document)

        # Sort by similarity score
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top_k unique results
        top_results = all_results[:self.top_k]

        logger.info(f"Retrieved {len(top_results)} unique documents using {len(perspectives)} query perspectives")
        return top_results

    @staticmethod
    def prepare_context(documents: List[Dict[str, Any]], max_tokens: int = 3800) -> str:
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
            doc_context = f"Document {i + 1}:\n"

            # Add any citation information
            if "citation" in doc["metadata"]:
                doc_context += f"Citation: {doc['metadata']['citation']}\n"

            # Add jurisdiction if available
            if "jurisdiction" in doc["metadata"]:
                doc_context += f"Jurisdiction: {doc['metadata']['jurisdiction']}\n"

            # Add document type if available
            if "document_type" in doc["metadata"]:
                doc_context += f"Document Type: {doc['metadata']['document_type']}\n"

            # Add perspective info
            if "perspective" in doc:
                doc_context += f"Retrieved using perspective: {doc['perspective']}\n"

            # Add the document text
            doc_context += f"Content:\n{doc['text']}\n\n"

            context_parts.append(doc_context)
            current_tokens += doc_tokens

        # Join all context parts
        full_context = "\n".join(context_parts)

        return full_context

    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a query through the multi-query retrieval pipeline.

        Args:
            query: The user's legal query

        Returns:
            Tuple of (context_for_llm, retrieved_documents)
        """
        logger.info(f"Processing query: '{query[:50]}...'")

        # Retrieve relevant documents using multiple perspectives
        documents = self.retrieve(query)

        # Prepare context for LLM
        context = self.prepare_context(documents)

        return context, documents

    #TODO: make the implementation use the combine_results method to combine results if using multi-query rag and show
    # the final results
    def combine_results(self, query_results: List[List[Dict[str, Any]]], perspectives: List[str]) -> List[
        Dict[str, Any]]:
        """
        Combine and deduplicate results from multiple queries.

        Args:
            query_results: List of result lists from different queries
            perspectives: List of query perspectives used

        Returns:
            Combined and deduplicated list of documents
        """
        # Create a dictionary to store documents by ID
        documents_by_id = {}

        # Process each result set
        for i, results in enumerate(query_results):
            perspective = perspectives[i] if i < len(perspectives) else "Unknown"

            for doc in results:
                doc_id = doc["id"]

                if doc_id in documents_by_id:
                    # Document already exists, update score
                    documents_by_id[doc_id]["score"] = max(documents_by_id[doc_id]["score"], doc["score"])
                    documents_by_id[doc_id]["perspectives"].append(perspective)
                else:
                    # New document, add to dictionary
                    doc["perspectives"] = [perspective]
                    documents_by_id[doc_id] = doc

        # Convert back to list and sort by score
        combined_results = list(documents_by_id.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top_k unique results
        top_results = combined_results[:self.top_k]

        logger.info(
            f"Combined {sum(len(results) for results in query_results)} documents into {len(top_results)} unique documents")
        return top_results
