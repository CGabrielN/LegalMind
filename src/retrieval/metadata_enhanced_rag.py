"""
LegalMind Metadata-Enhanced Retrieval Strategy

This module implements metadata-enhanced retrieval that uses jurisdiction filtering,
court hierarchy weighting, recency analysis, and citation network mapping to
improve retrieval quality for legal documents.
"""

import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetadataEnhancedRAG:
    """
    Implements metadata-enhanced retrieval for legal documents.
    """

    def __init__(self, resource_manager=None):
        """Initialize the metadata-enhanced retrieval system."""
        if resource_manager is None:
            from src.core.resource_manager import ResourceManager
            resource_manager = ResourceManager()
        self.resource_manager = resource_manager
        self.config = resource_manager.config

        self.vector_store = resource_manager.vector_store
        self.embedding_model = resource_manager.embedding_model
        self.query_expander = resource_manager.query_expander
        self.top_k = self.config["rag"]["metadata_enhanced"]["top_k"]

        # Australian court hierarchy (higher number = higher court)
        self.court_hierarchy = {
            "HCA": 10,  # High Court of Australia
            "FCAFC": 9,  # Federal Court of Australia (Full Court)
            "FCA": 8,  # Federal Court of Australia
            "NSWCA": 7,  # NSW Court of Appeal
            "VSCA": 7,  # Victorian Court of Appeal
            "QCA": 7,  # Queensland Court of Appeal
            "WASCA": 7,  # WA Court of Appeal
            "NSWSC": 6,  # NSW Supreme Court
            "VSC": 6,  # Victorian Supreme Court
            "QSC": 6,  # Queensland Supreme Court
            "WASC": 6,  # WA Supreme Court
            "SASC": 6,  # SA Supreme Court
            "TASSC": 6,  # Tasmanian Supreme Court
            "NTSC": 6,  # NT Supreme Court
            "ACTSC": 6,  # ACT Supreme Court
            "NSWDC": 5,  # NSW District Court
            "VCC": 5,  # Victorian County Court
            "QDC": 5,  # Queensland District Court
            "WADC": 5,  # WA District Court
            "SADC": 5,  # SA District Court
            "NSWLC": 4,  # NSW Local Court
            "VMC": 4,  # Victorian Magistrates Court
            "QMC": 4,  # Queensland Magistrates Court
            "FCCA": 7,  # Federal Circuit Court of Australia
            "FMC": 7  # Federal Magistrates Court (older name for FCCA)
        }

        logger.info("Initialized Metadata-Enhanced RAG system")

    @staticmethod
    def _extract_court_from_citation(citation: str) -> Optional[str]:
        """
        Extract court identifier from a legal citation.

        Args:
            citation: Legal citation string

        Returns:
            Court identifier or None
        """
        # Pattern for Australian citations
        pattern = r"\[\d{4}\]\s+([A-Z]+)\s+\d+"
        match = re.search(pattern, citation)

        if match:
            return match.group(1)

        return None

    @staticmethod
    def _extract_year_from_citation(citation: str) -> Optional[int]:
        """
        Extract year from a legal citation.

        Args:
            citation: Legal citation string

        Returns:
            Year as integer or None
        """
        # Pattern for year in Australian citations
        pattern = r"\[(\d{4})\]"
        match = re.search(pattern, citation)

        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None

        return None

    @staticmethod
    def _calculate_recency_score(year: Optional[int]) -> float:
        """
        Calculate recency score based on document year.

        Args:
            year: Document year

        Returns:
            Recency score (0-1)
        """
        if year is None:
            return 0.5  # Neutral score for unknown year

        current_year = datetime.now().year
        years_ago = current_year - year

        # Exponential decay function
        if years_ago <= 0:
            return 1.0  # Current year gets perfect score
        elif years_ago > 50:
            return 0.1  # Very old cases get low but non-zero score
        else:
            # Score from 0.1 to 1.0 with exponential decay
            return max(0.1, 1.0 * (0.95 ** years_ago))

    def _calculate_court_hierarchy_score(self, court: Optional[str]) -> float:
        """
        Calculate court hierarchy score.

        Args:
            court: Court identifier

        Returns:
            Court hierarchy score (0-1)
        """
        if court is None:
            return 0.5  # Neutral score for unknown court

        # Get hierarchy level
        hierarchy_level = self.court_hierarchy.get(court, 0)

        # Normalize to 0-1 scale
        max_level = max(self.court_hierarchy.values())
        return hierarchy_level / max_level

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents using metadata-enhanced retrieval.

        Args:
            query: The user's legal query

        Returns:
            List of relevant documents with text and metadata
        """
        logger.info(f"Metadata-enhanced retrieval for: '{query[:50]}...'")

        # Identify jurisdiction
        jurisdiction = self.query_expander.identify_jurisdiction(query)

        # Basic similarity search
        query_embedding = self.embedding_model.embed_text(query)

        # Set up filters
        filters = {}
        if jurisdiction:
            filters["jurisdiction"] = jurisdiction

        # Query with filters if applicable
        if filters:
            results = self.vector_store.metadata_filter_query(query, filters, n_results=self.top_k * 2)
        else:
            results = self.vector_store.query(query, n_results=self.top_k * 2)

        # Process results and add metadata scores
        documents = []
        for i in range(len(results["documents"][0])):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"][0] else {}
            doc_id = results["ids"][0][i] if results["ids"][0] else f"doc_{i}"

            # Get semantic similarity score
            similarity_score = 1.0 - results["distances"][0][i] if results["distances"][0] else 0.0

            # Extract citation if available
            citation = metadata.get("citation", "")

            # Extract court and year
            court = self._extract_court_from_citation(citation)
            year = self._extract_year_from_citation(citation)

            # Calculate metadata scores
            court_score = self._calculate_court_hierarchy_score(court)
            recency_score = self._calculate_recency_score(year)

            # Calculate combined score
            # Weights can be adjusted based on importance of each factor
            combined_score = (
                    0.6 * similarity_score +  # Semantic similarity is most important
                    0.2 * court_score +  # Court hierarchy
                    0.2 * recency_score  # Recency
            )

            document = {
                "text": text,
                "metadata": metadata,
                "id": doc_id,
                "similarity_score": similarity_score,
                "court_score": court_score,
                "recency_score": recency_score,
                "combined_score": combined_score,
                "court": court,
                "year": year
            }

            documents.append(document)

        # Sort by combined score
        documents.sort(key=lambda x: x["combined_score"], reverse=True)

        # Take top_k
        top_documents = documents[:self.top_k]

        logger.info(f"Retrieved {len(top_documents)} documents with metadata enhancement")
        return top_documents

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

            # Add year if available
            if "year" in doc and doc["year"]:
                doc_context += f"Year: {doc['year']}\n"

            # Add court if available
            if "court" in doc and doc["court"]:
                doc_context += f"Court: {doc['court']}\n"

            # Add relevance information
            doc_context += f"Relevance: {doc['combined_score']:.2f}\n"

            # Add the document text
            doc_context += f"Content:\n{doc['text']}\n\n"

            context_parts.append(doc_context)
            current_tokens += doc_tokens

        # Join all context parts
        full_context = "\n".join(context_parts)

        return full_context

    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a query through the metadata-enhanced retrieval pipeline.

        Args:
            query: The user's legal query

        Returns:
            Tuple of (context_for_llm, retrieved_documents)
        """
        logger.info(f"Processing query with metadata enhancement: '{query[:50]}...'")

        # Retrieve relevant documents with metadata enhancement
        documents = self.retrieve(query)

        # Prepare context for LLM
        context = self.prepare_context(documents)

        return context, documents

    @staticmethod
    def build_citation_network(documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Build a citation network based on retrieved documents.

        Args:
            documents: List of retrieved documents

        Returns:
            Dictionary mapping document IDs to lists of cited document IDs
        """
        # Extract citations from documents
        citation_network = {}
        citation_pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"

        # Create mapping from citation to document ID
        citation_to_id = {}
        for doc in documents:
            citation = doc["metadata"].get("citation", "")
            if citation:
                citation_to_id[citation] = doc["id"]

        # Build the network
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]

            # Find all citations in the text
            citations = re.findall(citation_pattern, text)

            # Map citations to document IDs where possible
            cited_ids = []
            for citation in citations:
                if citation in citation_to_id:
                    cited_id = citation_to_id[citation]
                    if cited_id != doc_id:  # Avoid self-citations
                        cited_ids.append(cited_id)

            # Add to network
            if cited_ids:
                citation_network[doc_id] = cited_ids

        return citation_network

    # TODO: use this so that rag can be used properly to get documents based on citation network
    def rerank_with_citation_network(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on citation network analysis.

        Args:
            documents: List of retrieved documents

        Returns:
            Reranked list of documents
        """
        # Build citation network
        citation_network = self.build_citation_network(documents)

        # Calculate citation scores (simplified PageRank)
        citation_scores = {doc["id"]: 1.0 for doc in documents}  # Initialize with equal scores

        # Perform a few iterations of PageRank
        for _ in range(3):
            new_scores = {doc_id: 0.15 for doc_id in citation_scores}  # Base score (damping factor)

            # Distribute scores through citations
            for citing_id, cited_ids in citation_network.items():
                if cited_ids:
                    score_to_distribute = 0.85 * citation_scores[citing_id] / len(cited_ids)
                    for cited_id in cited_ids:
                        if cited_id in new_scores:
                            new_scores[cited_id] += score_to_distribute

            # Normalize scores
            max_score = max(new_scores.values()) if new_scores else 1.0
            citation_scores = {doc_id: score / max_score for doc_id, score in new_scores.items()}

        # Update document scores
        for doc in documents:
            doc_id = doc["id"]
            citation_score = citation_scores.get(doc_id, 0.0)

            # Combine with previous score
            doc["citation_score"] = citation_score
            doc["combined_score"] = (
                    0.7 * doc["combined_score"] +  # Original combined score
                    0.3 * citation_score  # Citation network score
            )

        # Rerank documents
        documents.sort(key=lambda x: x["combined_score"], reverse=True)

        return documents
