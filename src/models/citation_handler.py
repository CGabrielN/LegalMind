"""
LegalMind Citation Handling

This module implements a comprehensive citation handling system for Australian
legal texts, providing extraction, validation, and cross-referencing against
Australian precedents.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a legal citation with its components."""
    full_citation: str
    case_name: str
    parties: List[str]
    year: str
    court: str
    court_abbr: str
    number: str
    jurisdiction: str
    citation_type: str = "case"  # case, statute, article
    valid: bool = True
    confidence: float = 1.0


class AustralianCitationExtractor:
    """
    Extracts and validates Australian legal citations from text.
    """

    def __init__(self):
        """Initialize the citation extractor."""
        # Australian case citation pattern
        # Example: "Smith v Jones [2010] NSWSC 123"
        self.case_pattern = r"([A-Za-z\s']+\sv\s[A-Za-z\s']+)\s\[(\d{4})\]\s([A-Z]+)\s(\d+)"

        # Australian statute citation pattern
        # Example: "Criminal Code Act 1995 (Cth)"
        self.statute_pattern = r"([A-Za-z\s]+Act\s\d{4})\s\(([A-Za-z]+)\)"

        # Court abbreviations to jurisdiction mapping
        self.court_jurisdictions = {
            "HCA": "High Court of Australia",
            "FCAFC": "Federal Court of Australia (Full Court)",
            "FCA": "Federal Court of Australia",
            "NSWSC": "New South Wales",
            "NSWCA": "New South Wales",
            "VSC": "Victoria",
            "VSCA": "Victoria",
            "QSC": "Queensland",
            "QCA": "Queensland",
            "WASC": "Western Australia",
            "WASCA": "Western Australia",
            "SASC": "South Australia",
            "SASCFC": "South Australia",
            "TASSC": "Tasmania",
            "NTSC": "Northern Territory",
            "ACTSC": "Australian Capital Territory"
        }

        # Statute jurisdiction abbreviations
        self.statute_jurisdictions = {
            "Cth": "Commonwealth",
            "NSW": "New South Wales",
            "Vic": "Victoria",
            "Qld": "Queensland",
            "WA": "Western Australia",
            "SA": "South Australia",
            "Tas": "Tasmania",
            "NT": "Northern Territory",
            "ACT": "Australian Capital Territory"
        }

        logger.info("Initialized Australian citation extractor")

    def extract_case_citations(self, text: str) -> List[Citation]:
        """
        Extract case citations from text.

        Args:
            text: Text to analyze

        Returns:
            List of Citation objects
        """
        citations = []

        try:
            # Clean up the text - remove extra spaces
            text = ' '.join(text.split())

            # Find all case citations in text
            matches = re.finditer(self.case_pattern, text)

            for match in matches:
                try:
                    case_name = match.group(1).strip()
                    year = match.group(2)
                    court_abbr = match.group(3)
                    number = match.group(4)

                    # Determine jurisdiction from court abbreviation
                    jurisdiction = self.court_jurisdictions.get(court_abbr, "Unknown")

                    # Create Citation object
                    citation = Citation(
                        full_citation=match.group(0).strip(),
                        case_name=case_name,
                        parties=case_name.split(" v "),
                        year=year,
                        court=self.court_jurisdictions.get(court_abbr, court_abbr),
                        court_abbr=court_abbr,
                        number=number,
                        jurisdiction=jurisdiction,
                        citation_type="case"
                    )

                    citations.append(citation)
                except Exception as e:
                    logger.warning(f"Error processing citation match: {str(e)}")
                    continue

            logger.info(f"Extracted {len(citations)} case citations")
            return citations
        except Exception as e:
            logger.error(f"Error in citation extraction: {str(e)}")
            return citations

    def extract_statute_citations(self, text: str) -> List[Citation]:
        """
        Extract statute citations from text.

        Args:
            text: Text to analyze

        Returns:
            List of Citation objects
        """
        citations = []

        # Find all statute citations in text
        matches = re.finditer(self.statute_pattern, text)

        for match in matches:
            statute_name = match.group(1)
            jurisdiction_abbr = match.group(2)

            # Extract year from statute name
            year_match = re.search(r"\d{4}", statute_name)
            year = year_match.group(0) if year_match else ""

            # Determine jurisdiction from abbreviation
            jurisdiction = self.statute_jurisdictions.get(jurisdiction_abbr, "Unknown")

            # Create Citation object
            citation = Citation(
                full_citation=match.group(0),
                case_name=statute_name,
                parties=[statute_name],
                year=year,
                court="",
                court_abbr="",
                number="",
                jurisdiction=jurisdiction,
                citation_type="statute"
            )

            citations.append(citation)

        logger.info(f"Extracted {len(citations)} statute citations")
        return citations

    def extract_all_citations(self, text: str) -> List[Citation]:
        """
        Extract all types of citations from text.

        Args:
            text: Text to analyze

        Returns:
            List of Citation objects
        """
        case_citations = self.extract_case_citations(text)
        statute_citations = self.extract_statute_citations(text)

        all_citations = case_citations + statute_citations
        logger.info(f"Extracted {len(all_citations)} total citations")

        return all_citations


class CitationFormatting:
    """
    Formats and styles legal citations for consistency.
    """

    def __init__(self):
        """Initialize citation formatter."""
        logger.info("Initialized citation formatter")

    def format_citation(self, citation: Citation) -> str:
        """
        Format a citation according to Australian legal style.

        Args:
            citation: Citation to format

        Returns:
            Formatted citation string
        """
        if citation.citation_type == "case":
            # Format case citation
            # Example: Smith v Jones [2010] NSWSC 123
            return f"{citation.case_name} [{citation.year}] {citation.court_abbr} {citation.number}"

        elif citation.citation_type == "statute":
            # Format statute citation
            # Example: Criminal Code Act 1995 (Cth)
            jurisdiction_abbr = next((abbr for abbr, full in self.get_jurisdiction_abbreviations().items()
                                      if full == citation.jurisdiction), "")

            return f"{citation.case_name} ({jurisdiction_abbr})"

        # Default: return original
        return citation.full_citation

    def format_all_citations(self, text: str) -> str:
        """
        Format all citations in a text for consistency.

        Args:
            text: Text containing citations

        Returns:
            Text with consistently formatted citations
        """
        # Extract citations
        extractor = AustralianCitationExtractor()
        citations = extractor.extract_all_citations(text)

        # Sort citations by position (to avoid replacing issues)
        citation_positions = [(c.full_citation, text.find(c.full_citation)) for c in citations]
        citation_positions.sort(key=lambda x: x[1], reverse=True)

        # Replace each citation with formatted version
        formatted_text = text
        for citation, pos in citation_positions:
            if pos >= 0:
                # Find matching Citation object
                for c in citations:
                    if c.full_citation == citation:
                        formatted_citation = self.format_citation(c)
                        formatted_text = formatted_text[:pos] + formatted_citation + formatted_text[
                                                                                     pos + len(citation):]
                        break

        return formatted_text

    @staticmethod
    def get_jurisdiction_abbreviations() -> Dict[str, str]:
        """Get mapping of jurisdiction abbreviations to full names."""
        return {
            "Cth": "Commonwealth",
            "NSW": "New South Wales",
            "Vic": "Victoria",
            "Qld": "Queensland",
            "WA": "Western Australia",
            "SA": "South Australia",
            "Tas": "Tasmania",
            "NT": "Northern Territory",
            "ACT": "Australian Capital Territory"
        }

    def add_citation_links(self, text: str) -> str:
        """
        Add hyperlinks to citations in HTML text.

        Args:
            text: HTML text with citations

        Returns:
            HTML text with linked citations
        """
        # Extract citations
        extractor = AustralianCitationExtractor()
        citations = extractor.extract_all_citations(text)

        # Sort citations by position (to avoid replacing issues)
        citation_positions = [(c.full_citation, text.find(c.full_citation)) for c in citations]
        citation_positions.sort(key=lambda x: x[1], reverse=True)

        # Replace each citation with linked version
        linked_text = text
        for citation, pos in citation_positions:
            if pos >= 0:
                # Find matching Citation object
                for c in citations:
                    if c.full_citation == citation:
                        link_url = self._generate_citation_link(c)
                        linked_citation = f'<a href="{link_url}" title="{c.full_citation}" class="legal-citation">{c.full_citation}</a>'
                        linked_text = linked_text[:pos] + linked_citation + linked_text[pos + len(citation):]
                        break

        return linked_text

    def _generate_citation_link(self, citation: Citation) -> str:
        """Generate a link URL for a citation."""
        if citation.citation_type == "case":
            # Example: https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWSC/2010/123.html
            court_path = citation.court_abbr.lower()
            jurisdiction_path = self._get_austlii_jurisdiction_path(citation.jurisdiction)

            return f"https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/{jurisdiction_path}/{court_path}/{citation.year}/{citation.number}.html"

        elif citation.citation_type == "statute":
            # Example: https://www.austlii.edu.au/cgi-bin/viewdb/au/legis/cth/consol_act/cca1995115/
            jurisdiction_path = self._get_austlii_jurisdiction_path(citation.jurisdiction)
            statute_path = citation.case_name.lower().replace(" ", "")

            return f"https://www.austlii.edu.au/cgi-bin/viewdb/au/legis/{jurisdiction_path}/consol_act/{statute_path}/"

        # Default: austlii search
        return f"https://www.austlii.edu.au/cgi-bin/sinosrch.cgi?query={citation.full_citation.replace(' ', '+')}"

    @staticmethod
    def _get_austlii_jurisdiction_path(jurisdiction: str) -> str:
        """Convert jurisdiction to AustLII path component."""
        mapping = {
            "Commonwealth": "cth",
            "New South Wales": "nsw",
            "Victoria": "vic",
            "Queensland": "qld",
            "Western Australia": "wa",
            "South Australia": "sa",
            "Tasmania": "tas",
            "Northern Territory": "nt",
            "Australian Capital Territory": "act"
        }

        return mapping.get(jurisdiction, "cth").lower()


class CitationVerification:
    """
    Verifies Australian legal citations against authoritative sources.
    """

    def __init__(self):
        """Initialize citation verifier."""
        self.extractor = AustralianCitationExtractor()
        logger.info("Initialized citation verifier")

    def verify_citation_in_context(self, citation: str, context: str) -> Tuple[bool, float]:
        """
        Verify if a citation appears in the context.

        Args:
            citation: Citation string
            context: Context text

        Returns:
            Tuple of (is_verified, confidence)
        """
        # Direct match is most reliable
        if citation in context:
            return True, 1.0

        # Extract citation components
        citations = self.extractor.extract_all_citations(citation + " " + context)
        if not citations:
            return False, 0.0

        # Find the target citation
        target = None
        for c in citations:
            if c.full_citation == citation:
                target = c
                break

        if not target:
            return False, 0.0

        # Check for partial matches in context
        # For cases, check if case name and year appear close together
        if target.citation_type == "case":
            if target.case_name in context and target.year in context:
                # Check proximity
                name_pos = context.find(target.case_name)
                year_pos = context.find(target.year)

                if abs(name_pos - year_pos) < 100:  # Within 100 chars
                    return True, 0.8

            # Check if both parties appear
            if len(target.parties) >= 2:
                party1, party2 = target.parties[0], target.parties[1]
                if party1 in context and party2 in context:
                    return True, 0.6

        # For statutes, check if name appears
        if target.citation_type == "statute":
            if target.case_name in context:
                return True, 0.7

        return False, 0.0

    def verify_all_citations_in_response(self, response: str, context: str) -> Dict[str, Any]:
        """
        Verify all citations in a response against the context.

        Args:
            response: Response text with citations
            context: Context text

        Returns:
            Dictionary with verification results
        """
        # Extract citations from response
        citations = self.extractor.extract_all_citations(response)

        # Verify each citation
        verification_results = []
        for citation in citations:
            verified, confidence = self.verify_citation_in_context(citation.full_citation, context)

            result = {
                "citation": citation.full_citation,
                "verified_in_context": verified,
                "context_confidence": confidence
            }

            verification_results.append(result)

        # Calculate overall verification score
        if verification_results:
            avg_confidence = sum(r["context_confidence"] for r in verification_results) / len(verification_results)
            verified_count = sum(1 for r in verification_results if r["verified_in_context"])
            verification_rate = verified_count / len(verification_results)
        else:
            avg_confidence = 1.0
            verification_rate = 1.0

        return {
            "citations": [c.full_citation for c in citations],
            "verification_results": verification_results,
            "avg_confidence": avg_confidence,
            "verification_rate": verification_rate,
            "has_unverified_citations": verification_rate < 1.0
        }

    def suggest_citations(self, text: str, context: str) -> List[Dict[str, Any]]:
        """
        Suggest citations from context that might be relevant to text.

        Args:
            text: Text to analyze
            context: Context containing potential citations

        Returns:
            List of suggested citations with relevance scores
        """
        # Extract citations from context
        context_citations = self.extractor.extract_all_citations(context)
        if not context_citations:
            return []

        # Extract legal topics from text
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP
        topics = self._extract_legal_topics(text)

        # Score citations by relevance to text
        suggestions = []
        for citation in context_citations:
            # Basic relevance scoring
            relevance = 0.0

            # Check if any citation parties appear in text
            for party in citation.parties:
                if party.lower() in text.lower():
                    relevance += 0.5

            # Check if any topics match
            context_snippet = self._extract_context_around_citation(citation.full_citation, context)
            context_topics = self._extract_legal_topics(context_snippet)

            matching_topics = set(topics) & set(context_topics)
            relevance += len(matching_topics) * 0.2

            # Only suggest if minimally relevant
            if relevance > 0.2:
                suggestions.append({
                    "citation": citation.full_citation,
                    "relevance": min(relevance, 1.0),
                    "context_snippet": context_snippet,
                    "matching_topics": list(matching_topics)
                })

        # Sort by relevance
        suggestions.sort(key=lambda x: x["relevance"], reverse=True)

        return suggestions

    @staticmethod
    def _extract_legal_topics(text: str) -> List[str]:
        """Extract legal topics from text."""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP

        legal_topics = {
            "contract": ["contract", "agreement", "offer", "acceptance", "consideration"],
            "tort": ["tort", "negligence", "duty of care", "breach", "damages"],
            "property": ["property", "land", "easement", "title", "possession"],
            "criminal": ["criminal", "crime", "offense", "charge", "guilty"],
            "constitutional": ["constitution", "constitutional", "commonwealth", "power"],
            "administrative": ["administrative", "review", "judicial review", "minister"],
            "family": ["family", "divorce", "custody", "marriage", "child"]
        }

        text_lower = text.lower()
        found_topics = []

        for topic, keywords in legal_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)

        return found_topics

    @staticmethod
    def _extract_context_around_citation(citation: str, context: str, window: int = 200) -> str:
        """Extract text around a citation."""
        position = context.find(citation)
        if position == -1:
            return ""

        start = max(0, position - window)
        end = min(len(context), position + len(citation) + window)

        return context[start:end]
