"""
LegalMind Text Preprocessing

This module handles text cleaning and preprocessing for legal documents.
"""

import re
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalTextPreprocessor:
    """
    Handles preprocessing of legal text documents.
    """

    def __init__(self):
        """Initialize the preprocessor."""
        logger.info("Initialized legal text preprocessor")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace, normalizing quotes, etc.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Fix common OCR errors in legal documents
        text = text.replace('1he', 'The').replace('0f', 'of')
        text = text.replace('Il', 'II').replace('I1', 'II')

        # Standardize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Clean up spaces around punctuation
        text = re.sub(r'\s+([.,;:?!])', r'\1', text)

        # Ensure proper spacing after punctuation
        text = re.sub(r'([.,;:?!])([^\s\d"])', r'\1 \2', text)

        # Fix spacing around brackets
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def standardize_citations(self, text: str) -> str:
        """
        Standardize legal citations format.

        Args:
            text: Text with citations

        Returns:
            Text with standardized citations
        """
        # Standardize "v" in case names (sometimes appears as "v.", "vs", "vs.", "v/s")
        text = re.sub(r'(\w+)\s+v[s./]+\s+(\w+)', r'\1 v \2', text)

        # Standardize year format [YEAR] in citations
        text = re.sub(r'\[\s*(\d{4})\s*\]', r'[\1]', text)

        # Standardize court abbreviations
        court_abbr = {
            'NSWSC': 'NSWSC',  # New South Wales Supreme Court
            'N.S.W.S.C.': 'NSWSC',
            'NSW. S.C.': 'NSWSC',
            'NSWCA': 'NSWCA',  # NSW Court of Appeal
            'N.S.W.C.A.': 'NSWCA',
            'FCA': 'FCA',  # Federal Court of Australia
            'F.C.A.': 'FCA',
            'FCAFC': 'FCAFC',  # Federal Court of Australia Full Court
            'F.C.A.F.C.': 'FCAFC',
            'HCA': 'HCA',  # High Court of Australia
            'H.C.A.': 'HCA'
        }

        for old, new in court_abbr.items():
            # Only replace if it's part of a citation (followed by a number)
            text = re.sub(rf'{re.escape(old)}\s+(\d+)', f'{new} \\1', text)

        return text

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from legal text.

        Args:
            text: Legal document text

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        # Extract jurisdiction
        jurisdiction_patterns = [
            (r'Supreme Court of New South Wales', 'New South Wales'),
            (r'Supreme Court of Victoria', 'Victoria'),
            (r'Supreme Court of Queensland', 'Queensland'),
            (r'Supreme Court of Western Australia', 'Western Australia'),
            (r'Supreme Court of South Australia', 'South Australia'),
            (r'Supreme Court of Tasmania', 'Tasmania'),
            (r'Supreme Court of the Northern Territory', 'Northern Territory'),
            (r'Supreme Court of the Australian Capital Territory', 'Australian Capital Territory'),
            (r'High Court of Australia', 'High Court of Australia'),
            (r'Federal Court of Australia', 'Federal Court of Australia')
        ]

        for pattern, jur in jurisdiction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                metadata['jurisdiction'] = jur
                break

        # Extract year
        year_match = re.search(r'\[\s*(\d{4})\s*\]', text)
        if year_match:
            metadata['year'] = year_match.group(1)

        # Extract citation
        citation_match = re.search(r'([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)', text)
        if citation_match:
            metadata['citation'] = citation_match.group(1)

        # Extract document type
        doc_types = {
            'judgment': ['judgment', 'judgement'],
            'decision': ['decision', 'ruling'],
            'order': ['order', 'decree'],
            'appeal': ['appeal'],
            'opinion': ['opinion']
        }

        for doc_type, keywords in doc_types.items():
            for keyword in keywords:
                if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                    metadata['document_type'] = doc_type.title()
                    break
            if 'document_type' in metadata:
                break

        return metadata

    def normalize_legal_text(self, text: str, extract_meta: bool = True) -> Dict[str, Any]:
        """
        Perform full normalization of legal text.

        Args:
            text: Raw legal text
            extract_meta: Whether to extract metadata

        Returns:
            Dictionary with normalized text and metadata
        """
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Standardize citations
        standardized_text = self.standardize_citations(cleaned_text)

        result = {
            "text": standardized_text
        }

        # Extract metadata if requested
        if extract_meta:
            metadata = self.extract_metadata(standardized_text)
            result["metadata"] = metadata

        return result

    def split_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split a legal document into its component sections.

        Args:
            text: Legal document text

        Returns:
            List of dictionaries with section text and type
        """
        # Common section headers in legal documents
        section_headers = [
            "INTRODUCTION", "BACKGROUND", "FACTS", "PROCEDURAL HISTORY",
            "ISSUES", "ARGUMENTS", "ANALYSIS", "REASONING", "DISCUSSION",
            "LAW", "CONCLUSION", "DECISION", "ORDER", "JUDGMENT", "ORDERS"
        ]

        # Create regex pattern for section detection
        header_pattern = "|".join(section_headers)
        matches = re.finditer(rf'(^|\n)(\d+\.\s*)?({header_pattern})[:\s]*(\n|$)', text, re.IGNORECASE)

        # Get match positions
        positions = [(m.start(), m.group(3)) for m in matches]
        if not positions:
            # If no sections found, return whole document as one section
            return [{"text": text, "section_type": "FULL_DOCUMENT"}]

        # Add end position
        positions.append((len(text), "END"))

        # Extract sections
        sections = []
        for i in range(len(positions) - 1):
            start_pos, section_type = positions[i]
            end_pos = positions[i + 1][0]

            # Extract section text
            section_text = text[start_pos:end_pos].strip()

            # Add to sections list
            sections.append({
                "text": section_text,
                "section_type": section_type.upper()
            })

        return sections

    def remove_boilerplate(self, text: str) -> str:
        """
        Remove standard legal boilerplate and headers.

        Args:
            text: Legal document text

        Returns:
            Text with boilerplate removed
        """
        # Remove standard headers
        text = re.sub(r'(?i)(IN THE SUPREME COURT OF.*?)\n', '', text)
        text = re.sub(r'(?i)(BEFORE:.*?)\n', '', text)
        text = re.sub(r'(?i)(File No:.*?)\n', '', text)
        text = re.sub(r'(?i)(Case No:.*?)\n', '', text)

        # Remove footers
        text = re.sub(r'(?i)(\d+ of \d+)', '', text)
        text = re.sub(r'(?i)(Page \d+ of \d+)', '', text)

        # Remove confidentiality notices
        text = re.sub(r'(?i)(CONFIDENTIAL.*?)((\n.+){0,3}\n)', '', text)

        return text.strip()

    def process_document(self, text: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline.

        Args:
            text: Raw legal document text

        Returns:
            Processed document with metadata
        """
        # Remove boilerplate
        cleaned = self.remove_boilerplate(text)

        # Normalize text
        normalized = self.normalize_legal_text(cleaned)

        # Split into sections
        sections = self.split_sections(normalized["text"])

        # Create final document
        document = {
            "text": normalized["text"],
            "metadata": normalized.get("metadata", {}),
            "sections": sections
        }

        return document