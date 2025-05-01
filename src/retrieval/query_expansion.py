"""
LegalMind Query Expansion Strategy

This module implements query expansion with legal terminology to enhance
retrieval by incorporating legal synonyms and jurisdiction-specific terms.
"""

import re
import yaml
import logging
from typing import List, Dict, Any, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class LegalQueryExpansion:
    """Expands queries with legal terminology for better retrieval."""

    def __init__(self):
        """Initialize the query expansion system."""
        # Legal terminology mappings (term -> synonyms/related concepts)
        self.legal_terminology = {
            # Negligence terms
            "negligence": ["duty of care", "breach of duty", "causation", "damages", "tort"],
            "duty of care": ["reasonable foreseeability", "proximity", "negligence"],
            "breach": ["standard of care", "reasonable person", "breach of duty"],

            # Contract terms
            "contract": ["agreement", "offer", "acceptance", "consideration", "contractual"],
            "offer": ["proposal", "tender", "invitation to treat"],
            "acceptance": ["assent", "agreement", "acquiescence"],
            "consideration": ["payment", "promise", "value", "quid pro quo"],

            # Property terms
            "property": ["real property", "land", "realty", "personal property"],
            "easement": ["right of way", "servitude", "encumbrance"],
            "title": ["ownership", "deed", "certificate of title", "interest"],

            # Criminal terms
            "criminal": ["offense", "crime", "illegal", "unlawful"],
            "murder": ["homicide", "manslaughter", "killing"],
            "theft": ["larceny", "stealing", "robbery", "burglary"],

            # Family law terms
            "divorce": ["dissolution", "separation", "marriage breakdown"],
            "custody": ["parental responsibility", "guardianship", "care arrangements"],

            # Employment terms
            "employment": ["work", "job", "labor", "workplace"],
            "unfair dismissal": ["wrongful termination", "unjust dismissal"],

            # Common legal procedures
            "appeal": ["review", "appellate", "challenge"],
            "evidence": ["proof", "testimony", "exhibit"],
            "damages": ["compensation", "restitution", "remedy"]
        }

        # Jurisdiction-specific terminology
        self.jurisdiction_terminology = {
            "nsw": {
                "court": ["NSW Supreme Court", "NSWSC", "NSW District Court", "Local Court"],
                "legislation": ["NSW legislation", "New South Wales statutes"]
            },
            "vic": {
                "court": ["Victorian Supreme Court", "VSC", "Victorian County Court", "Magistrates' Court"],
                "legislation": ["Victorian legislation", "Victoria statutes"]
            },
            "qld": {
                "court": ["Queensland Supreme Court", "QSC", "Queensland District Court"],
                "legislation": ["Queensland legislation", "Queensland statutes"]
            },
            "wa": {
                "court": ["Western Australia Supreme Court", "WASC", "WA District Court"],
                "legislation": ["Western Australia legislation", "WA statutes"]
            },
            "sa": {
                "court": ["South Australia Supreme Court", "SASC", "SA District Court"],
                "legislation": ["South Australia legislation", "SA statutes"]
            },
            "tas": {
                "court": ["Tasmania Supreme Court", "TASSC", "Tasmanian Magistrates Court"],
                "legislation": ["Tasmania legislation", "Tasmanian statutes"]
            },
            "act": {
                "court": ["ACT Supreme Court", "ACTSC", "ACT Magistrates Court"],
                "legislation": ["ACT legislation", "Australian Capital Territory statutes"]
            },
            "nt": {
                "court": ["Northern Territory Supreme Court", "NTSC", "NT Local Court"],
                "legislation": ["Northern Territory legislation", "NT statutes"]
            },
            "commonwealth": {
                "court": ["High Court of Australia", "HCA", "Federal Court", "FCA", "Federal Circuit Court"],
                "legislation": ["Commonwealth legislation", "federal statutes", "Australian legislation"]
            }
        }

        logger.info("Initialized legal query expansion system")

    def identify_legal_terms(self, query: str) -> List[str]:
        """
        Identify legal terms in the query.

        Args:
            query: The user's query

        Returns:
            List of identified legal terms
        """
        query_lower = query.lower()
        identified_terms = []

        # Check for each term in our dictionary
        for term in self.legal_terminology.keys():
            # Use word boundary regex to match whole words
            if re.search(r'\b' + re.escape(term) + r'\b', query_lower):
                identified_terms.append(term)

        logger.info(f"Identified legal terms: {identified_terms}")
        return identified_terms

    def identify_jurisdiction(self, query: str) -> str:
        """
        Identify jurisdiction mentioned in the query.

        Args:
            query: The user's query

        Returns:
            Identified jurisdiction or empty string
        """
        query_lower = query.lower()

        # Check for explicit jurisdiction mentions
        jurisdiction_keywords = {
            "nsw": ["nsw", "new south wales", "sydney"],
            "vic": ["vic", "victoria", "melbourne"],
            "qld": ["qld", "queensland", "brisbane"],
            "wa": ["wa", "western australia", "perth"],
            "sa": ["sa", "south australia", "adelaide"],
            "tas": ["tas", "tasmania", "hobart"],
            "act": ["act", "australian capital territory", "canberra"],
            "nt": ["nt", "northern territory", "darwin"],
            "commonwealth": ["commonwealth", "federal", "australia", "australian"]
        }

        for jurisdiction, keywords in jurisdiction_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    logger.info(f"Identified jurisdiction: {jurisdiction}")
                    return jurisdiction

        return ""

    def expand_query(self, query: str) -> List[str]:
        """
        Expand the query with legal terminology.

        Args:
            query: The user's query

        Returns:
            List of expanded queries
        """
        # Start with the original query
        expanded_queries = [query]

        # Identify legal terms
        legal_terms = self.identify_legal_terms(query)

        # Identify jurisdiction
        jurisdiction = self.identify_jurisdiction(query)

        # Expand with synonyms for each legal term
        for term in legal_terms:
            synonyms = self.legal_terminology.get(term, [])

            for synonym in synonyms:
                # Create expanded query by replacing the term with its synonym
                expanded_query = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    synonym,
                    query,
                    flags=re.IGNORECASE
                )

                if expanded_query != query and expanded_query not in expanded_queries:
                    expanded_queries.append(expanded_query)

        # Add jurisdiction-specific terminology if jurisdiction identified
        if jurisdiction:
            jurisdiction_terms = self.jurisdiction_terminology.get(jurisdiction, {})

            for category, terms in jurisdiction_terms.items():
                # Take the first term in each category
                if terms:
                    # Create jurisdiction-specific query
                    jurisdiction_query = f"{query} {terms[0]}"
                    if jurisdiction_query not in expanded_queries:
                        expanded_queries.append(jurisdiction_query)

        logger.info(f"Expanded queries: {expanded_queries}")
        return expanded_queries

    def expand_with_legal_context(self, query: str) -> List[Tuple[str, float]]:
        """
        Expand the query with legal context and assign weights.

        Args:
            query: The user's query

        Returns:
            List of (expanded_query, weight) tuples
        """
        # Start with the original query (weight 1.0)
        expanded_queries = [(query, 1.0)]

        # Identify legal terms
        legal_terms = self.identify_legal_terms(query)

        # Identify jurisdiction
        jurisdiction = self.identify_jurisdiction(query)

        # Expand with synonyms for each legal term
        for term in legal_terms:
            synonyms = self.legal_terminology.get(term, [])

            for synonym in synonyms:
                # Create expanded query by replacing the term with its synonym
                expanded_query = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    synonym,
                    query,
                    flags=re.IGNORECASE
                )

                if expanded_query != query and expanded_query not in [q[0] for q in expanded_queries]:
                    # Use slightly lower weight for synonym replacements
                    expanded_queries.append((expanded_query, 0.8))

        # Add jurisdiction-specific terminology if jurisdiction identified
        if jurisdiction:
            jurisdiction_terms = self.jurisdiction_terminology.get(jurisdiction, {})

            for category, terms in jurisdiction_terms.items():
                # Take the first term in each category
                if terms:
                    # Create jurisdiction-specific query
                    jurisdiction_query = f"{query} {terms[0]}"
                    if jurisdiction_query not in [q[0] for q in expanded_queries]:
                        # Use high weight for jurisdiction-specific queries
                        expanded_queries.append((jurisdiction_query, 0.9))

        # Add legal context phrases for common legal queries
        if "negligence" in query.lower():
            expanded_queries.append((f"{query} elements of negligence australia", 0.7))
        elif "contract" in query.lower():
            expanded_queries.append((f"{query} elements of contract formation australia", 0.7))
        elif "criminal" in query.lower() or "crime" in query.lower():
            expanded_queries.append((f"{query} criminal elements burden of proof", 0.7))

        logger.info(f"Expanded queries with weights: {expanded_queries}")
        return expanded_queries