"""
LegalMind Metadata Filtering

This module implements metadata-based filtering for legal document retrieval,
allowing users to filter by jurisdiction, document type, and other metadata.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Handles metadata-based filtering for legal document retrieval.
    """

    def __init__(self, resource_manager=None):
        """Initialize the metadata filter."""
        if resource_manager is None:
            from src.core.resource_manager import ResourceManager
            resource_manager = ResourceManager()
        self.resource_manager = resource_manager
        self.config = resource_manager.config
        self.vector_store = resource_manager.vector_store

        # Australian jurisdictions
        self.jurisdictions = {
            "nsw": "New South Wales",
            "vic": "Victoria",
            "qld": "Queensland",
            "wa": "Western Australia",
            "sa": "South Australia",
            "tas": "Tasmania",
            "nt": "Northern Territory",
            "act": "Australian Capital Territory",
            "cth": "Commonwealth",
            "federal": "Federal",
            "hca": "High Court of Australia"
        }

        # Document types
        self.document_types = {
            "decision": "Decision",
            "order": "Order",
            "judgment": "Judgment",
            "opinion": "Opinion",
            "appeal": "Appeal"
        }

        logger.info("Initialized metadata filter")

    def extract_jurisdiction_from_query(self, query: str) -> Optional[str]:
        """
        Extract jurisdiction information from query text.

        Args:
            query: The user's query

        Returns:
            Extracted jurisdiction or None
        """
        query_lower = query.lower()

        # Check for jurisdiction names in query
        for key, name in self.jurisdictions.items():
            # Check for full name
            if name.lower() in query_lower:
                return name

            # Check for abbreviation
            if f" {key} " in f" {query_lower} ":
                return name

        # Check for common patterns
        patterns = [
            r"in\s+(new\s+south\s+wales|victoria|queensland|western\s+australia|south\s+australia|tasmania|northern\s+territory|act)",
            r"under\s+(nsw|vic|qld|wa|sa|tas|nt|act|cth|federal)\s+law",
            r"(nsw|vic|qld|wa|sa|tas|nt|act|cth|federal)\s+case"
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                jurisdiction_text = match.group(1)

                # Map abbreviation to full name
                for key, name in self.jurisdictions.items():
                    if key == jurisdiction_text.lower():
                        return name

                # Return as is if full name
                return jurisdiction_text.title()

        return None

    def extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract filter conditions from query.

        Args:
            query: The user's query

        Returns:
            Dictionary of filter conditions
        """
        filters = {}

        # Extract jurisdiction
        jurisdiction = self.extract_jurisdiction_from_query(query)
        if jurisdiction:
            filters["jurisdiction"] = jurisdiction

        # Extract document type
        doc_type_pattern = r"(decision|judgment|judgement|order|opinion|appeal)"
        doc_type_match = re.search(doc_type_pattern, query.lower())
        if doc_type_match:
            doc_type = doc_type_match.group(1)
            # Normalize "judgement" to "judgment"
            if doc_type == "judgement":
                doc_type = "judgment"
            filters["document_type"] = doc_type.title()

        # Extract case citation if present
        citation_pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"
        citation_match = re.search(citation_pattern, query)
        if citation_match:
            filters["citation"] = citation_match.group(1)

        # Extract year if present
        year_pattern = r"\b(19\d{2}|20\d{2})\b"
        year_match = re.search(year_pattern, query)
        if year_match:
            filters["year"] = year_match.group(1)

        logger.info(f"Extracted filters from query: {filters}")
        return filters

    def get_available_metadata_values(self, field: str) -> Set[str]:
        """
        Get all available values for a metadata field from the vector store.

        Args:
            field: Metadata field name

        Returns:
            Set of unique values for the field
        """
        # This is a placeholder - in a real implementation, this would query
        # the vector store to get unique values for a field.
        # For now, we'll return predefined values for known fields.

        if field == "jurisdiction":
            return set(self.jurisdictions.values())
        elif field == "document_type":
            return set(self.document_types.values())
        else:
            return set()

    def filter_query(self, query: str, explicit_filters: Optional[Dict[str, Any]] = None) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform a filtered query using metadata.

        Args:
            query: The user's query
            explicit_filters: Optional explicit filters to use instead of extracting

        Returns:
            Tuple of (filtered_documents, applied_filters)
        """
        # Get filters (either explicit or extracted from query)
        filters = explicit_filters or self.extract_filters_from_query(query)

        # Query the vector store with filters
        results = self.vector_store.metadata_filter_query(
            query_text=query,
            filters=filters,
            n_results=self.config["rag"]["basic"]["top_k"]
        )

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

        return documents, filters
