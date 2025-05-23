"""
LegalMind Advanced RAG Integration

This module integrates all advanced RAG strategies (query expansion, multi-query,
and metadata-enhanced retrieval) into a unified retrieval system.
"""

import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedRAG:
    """
    Integrates multiple RAG strategies for comprehensive legal retrieval.
    """

    def __init__(self, resource_manager = None):
        """Initialize the advanced RAG system."""
        if resource_manager is None:
            from src.core.resource_manager import ResourceManager
            resource_manager = ResourceManager()
        self.resource_manager = resource_manager
        self.config = self.resource_manager.config

        # Initialize component RAG strategies
        self.basic_rag = self.resource_manager.basic_rag
        self.query_expansion = self.resource_manager.query_expander
        self.multi_query_rag = self.resource_manager.multi_query_rag
        self.metadata_enhanced_rag = self.resource_manager.metadata_enhanced_rag

        self.top_k = self.config["rag"]["advanced"]["top_k"]
        logger.info("Initialized Advanced RAG system")

    def determine_best_strategy(self, query: str) -> str:
        """
        Determine the best RAG strategy for a given query.

        Args:
            query: The user's legal query

        Returns:
            Strategy name: 'basic', 'query_expansion', 'multi_query', or 'metadata_enhanced'
        """
        query_lower = query.lower()

        # Check for jurisdiction-specific queries
        jurisdiction = self.query_expansion.identify_jurisdiction(query)
        if jurisdiction:
            logger.info(f"Selected metadata-enhanced strategy due to jurisdiction: {jurisdiction}")
            return "metadata_enhanced"

        # Check for multi-perspective queries
        perspective_indicators = ["plaintiff", "defendant", "prosecution", "defense",
                                  "compare", "contrast", "different", "perspectives"]
        if any(indicator in query_lower for indicator in perspective_indicators):
            logger.info("Selected multi-query strategy due to perspective indicators")
            return "multi_query"

        # Check for specialized legal terminology
        legal_terms = self.query_expansion.identify_legal_terms(query)
        if len(legal_terms) >= 2:
            logger.info(f"Selected query expansion strategy due to legal terms: {legal_terms}")
            return "query_expansion"

        # Default to basic RAG
        logger.info("Selected basic RAG strategy (default)")
        return "basic"

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents using the most appropriate strategy.

        Args:
            query: The user's legal query

        Returns:
            List of relevant documents with text and metadata
        """
        # Determine best strategy
        strategy = self.determine_best_strategy(query)

        # Retrieve using appropriate strategy
        if strategy == "metadata_enhanced":
            logger.info("Using Metadata-Enhanced retrieval")
            documents = self.metadata_enhanced_rag.retrieve(query)
        elif strategy == "multi_query":
            logger.info("Using Multi-Query retrieval")
            documents = self.multi_query_rag.retrieve(query)
        elif strategy == "query_expansion":
            logger.info("Using Query Expansion retrieval")

            # Expand the query
            expanded_queries = self.query_expansion.expand_query(query)

            # Use first expanded query for retrieval
            expanded_query = expanded_queries[0] if expanded_queries else query
            logger.info(f"Expanded query: {expanded_query}")

            # Use basic RAG with expanded query
            _, documents = self.basic_rag.process_query(expanded_query)
        else:
            logger.info("Using Basic RAG retrieval")
            _, documents = self.basic_rag.process_query(query)

        return documents

    def retrieve_with_all_strategies(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents using all strategies for comparison.

        Args:
            query: The user's legal query

        Returns:
            Dictionary mapping strategy names to document lists
        """
        results = {}

        # Basic RAG (always run as baseline)
        _, basic_docs = self.basic_rag.process_query(query)
        results["basic"] = basic_docs

        # Query Expansion
        expanded_queries = self.query_expansion.expand_query(query)
        expanded_query = expanded_queries[0] if expanded_queries else query
        _, query_exp_docs = self.basic_rag.process_query(expanded_query)
        results["query_expansion"] = query_exp_docs

        # Multi-Query
        multi_query_docs = self.multi_query_rag.retrieve(query)
        results["multi_query"] = multi_query_docs

        # Metadata-Enhanced
        metadata_docs = self.metadata_enhanced_rag.retrieve(query)
        results["metadata_enhanced"] = metadata_docs

        return results

    def prepare_context(self, documents: List[Dict[str, Any]], max_tokens: int = 3800) -> str:
        """
        Prepare retrieved documents as context for the LLM.

        Args:
            documents: List of retrieved documents
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        return self.basic_rag.prepare_context(documents, max_tokens)

    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a query through the advanced RAG pipeline.

        Args:
            query: The user's legal query

        Returns:
            Tuple of (context_for_llm, retrieved_documents)
        """
        logger.info(f"Processing query with advanced RAG: '{query[:50]}...'")

        # Retrieve documents with best strategy
        documents = self.retrieve(query)

        # Prepare context for LLM
        context = self.prepare_context(documents)

        return context, documents

    def explain_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Explain the retrieval process and results for a query.

        Args:
            query: The user's legal query

        Returns:
            Dictionary with retrieval explanation
        """
        # Determine best strategy
        strategy = self.determine_best_strategy(query)

        # Get results from multiple strategies if enabled
        results = self.retrieve_with_all_strategies(query)

        # Prepare explanation
        explanation = {
            "query": query,
            "best_strategy": strategy,
            "results_by_strategy": {}
        }

        # Add result summaries for each strategy
        for strat, docs in results.items():
            explanation["results_by_strategy"][strat] = {
                "num_docs": len(docs),
                "top_citations": [doc["metadata"].get("citation", "") for doc in docs[:3] if "metadata" in doc]
            }

        # Add strategy-specific details
        if strategy == "query_expansion":
            expanded_queries = self.query_expansion.expand_query(query)
            explanation["query_expansion"] = {
                "original_query": query,
                "expanded_queries": expanded_queries,
                "legal_terms": self.query_expansion.identify_legal_terms(query)
            }
        elif strategy == "multi_query":
            explanation["multi_query"] = {
                "perspectives": self.multi_query_rag.generate_query_perspectives(query)
            }
        elif strategy == "metadata_enhanced":
            explanation["metadata_enhanced"] = {
                "jurisdiction": self.query_expansion.identify_jurisdiction(query)
            }

        return explanation
