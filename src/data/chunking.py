"""
LegalMind Document Chunking Strategies

This module implements specialized chunking strategies for legal documents,
with a focus on Australian legal texts. It preserves legal context,
citations, and structures documents in a way that maintains their
interpretability.
"""

import re
import yaml
import logging
import tiktoken
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get chunking configuration
CHUNK_SIZES = config["chunking"]["chunk_sizes"]

@dataclass
class Chunk:
    """Represents a chunk of a legal document with metadata."""
    text: str
    metadata: Dict[str, Any]
    tokens: int

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text using tiktoken."""
    encoder = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding, widely compatible
    return len(encoder.encode(text))

def extract_citation(text: str) -> Optional[str]:
    """Extract Australian legal citation from text."""
    # Pattern for Australian citations like "Nasr v NRMA Insurance [2006] NSWSC 1918"
    pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def identify_document_type(text: str) -> str:
    """Identify the type of legal document."""
    if re.search(r"(decision|judgment|judgement|opinion)", text.lower()):
        return "Decision"
    elif re.search(r"(order|decree)", text.lower()):
        return "Order"
    elif re.search(r"(appeal|application)", text.lower()):
        return "Appeal"
    else:
        return "Other"

def identify_jurisdiction(text: str) -> Optional[str]:
    """Identify the Australian jurisdiction from text."""
    # Look for court identifiers in citations or text
    jurisdictions = {
        "NSWSC": "New South Wales Supreme Court",
        "NSWCA": "New South Wales Court of Appeal",
        "VSC": "Victoria Supreme Court",
        "QSC": "Queensland Supreme Court",
        "WASC": "Western Australia Supreme Court",
        "SASC": "South Australia Supreme Court",
        "TASSC": "Tasmania Supreme Court",
        "NTSC": "Northern Territory Supreme Court",
        "ACTSC": "Australian Capital Territory Supreme Court",
        "HCA": "High Court of Australia",
        "FCA": "Federal Court of Australia",
        "FCAFC": "Federal Court of Australia Full Court"
    }

    for code, full_name in jurisdictions.items():
        if code in text:
            return full_name

    # Try to find jurisdiction in text
    states = ["New South Wales", "Victoria", "Queensland", "Western Australia",
              "South Australia", "Tasmania", "Northern Territory", "Australian Capital Territory"]
    for state in states:
        if state in text:
            return state

    return None

def extract_section_type(text: str) -> str:
    """Identify the type of legal section (facts, arguments, holding)."""
    text_lower = text.lower()

    # Check for factual background sections
    if any(term in text_lower for term in ["background", "facts", "factual", "circumstances"]):
        return "factual_backgrounds"

    # Check for legal reasoning sections
    elif any(term in text_lower for term in ["reasoning", "analysis", "consideration", "discussed", "principles"]):
        return "legal_reasoning"

    # Check for holdings/decisions
    elif any(term in text_lower for term in ["holding", "conclusion", "decision", "order", "judgment", "therefore"]):
        return "case_holdings"

    # Default to legal_reasoning if unclear
    return "legal_reasoning"

def chunk_document_by_case(document: str, metadata: Dict[str, Any]) -> List[Chunk]:
    """Top-level chunking by case document."""
    # For full case documents, we keep them together to preserve full context
    return [Chunk(
        text=document,
        metadata=metadata,
        tokens=count_tokens(document)
    )]

def chunk_document_by_sections(document: str, metadata: Dict[str, Any]) -> List[Chunk]:
    """Mid-level chunking by legal reasoning sections."""
    chunks = []

    # Simple section detection based on common legal document structure
    # More sophisticated approaches could use legal NLP models or regex patterns
    section_markers = [
        r"BACKGROUND",
        r"FACTS",
        r"ISSUES?",
        r"ARGUMENTS?",
        r"REASONING",
        r"ANALYSIS",
        r"DECISION",
        r"CONCLUSION",
        r"ORDER",
        r"JUDGMENT"
    ]

    # Create regex pattern to split by section headers
    pattern = f"({'|'.join(section_markers)})"
    sections = re.split(f"({pattern})", document, flags=re.IGNORECASE)

    # Process sections
    current_section = ""

    for i, section in enumerate(sections):
        if i % 2 == 1:  # This is a section header
            current_section = section.strip()
            continue

        if current_section and section.strip():
            # Determine section type for token sizing
            section_type = extract_section_type(current_section + section)

            # Get token limits for this section type
            min_tokens = CHUNK_SIZES.get(section_type, {}).get("min_tokens", 400)
            max_tokens = CHUNK_SIZES.get(section_type, {}).get("max_tokens", 600)

            section_text = current_section + "\n" + section.strip()
            section_tokens = count_tokens(section_text)

            # Create section metadata
            section_metadata = metadata.copy()
            section_metadata["section_type"] = section_type
            section_metadata["section_header"] = current_section

            # Extract any citation in this section
            citation = extract_citation(section_text)
            if citation:
                section_metadata["citation"] = citation

            # If section is too large, break it down further
            if section_tokens > max_tokens:
                sub_chunks = chunk_by_paragraphs(section_text, section_metadata, min_tokens, max_tokens)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=section_text,
                    metadata=section_metadata,
                    tokens=section_tokens
                ))

    return chunks

def chunk_by_paragraphs(text: str, metadata: Dict[str, Any], min_tokens: int, max_tokens: int) -> List[Chunk]:
    """Paragraph-level chunking for specific legal points."""
    chunks = []
    paragraphs = re.split(r"\n\s*\n", text)

    current_chunk = ""
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_tokens = count_tokens(paragraph)

        # If paragraph alone exceeds max tokens, we need to split it by sentences
        if paragraph_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sentence_chunks = []
            current_sentence_chunk = ""
            current_sentence_tokens = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_tokens = count_tokens(sentence)

                # If adding this sentence would exceed max tokens, create a new chunk
                if current_sentence_tokens + sentence_tokens > max_tokens and current_sentence_chunk:
                    sentence_chunks.append((current_sentence_chunk, current_sentence_tokens))
                    current_sentence_chunk = sentence
                    current_sentence_tokens = sentence_tokens
                else:
                    current_sentence_chunk += " " + sentence if current_sentence_chunk else sentence
                    current_sentence_tokens += sentence_tokens

            # Add the last sentence chunk if it exists
            if current_sentence_chunk:
                sentence_chunks.append((current_sentence_chunk, current_sentence_tokens))

            # Add sentence chunks to our paragraph chunks
            for sentence_chunk, sentence_tokens in sentence_chunks:
                # If we can add to current chunk, do so
                if current_tokens + sentence_tokens <= max_tokens:
                    current_chunk += "\n\n" + sentence_chunk if current_chunk else sentence_chunk
                    current_tokens += sentence_tokens
                else:
                    # If current chunk is substantial enough, add it
                    if current_tokens >= min_tokens:
                        # Create metadata for this chunk
                        chunk_metadata = metadata.copy()
                        citation = extract_citation(current_chunk)
                        if citation:
                            chunk_metadata["citation"] = citation

                        chunks.append(Chunk(
                            text=current_chunk,
                            metadata=chunk_metadata,
                            tokens=current_tokens
                        ))

                    # Start a new chunk with this sentence
                    current_chunk = sentence_chunk
                    current_tokens = sentence_tokens

        # If we can add this paragraph to current chunk, do so
        elif current_tokens + paragraph_tokens <= max_tokens:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            current_tokens += paragraph_tokens
        else:
            # If current chunk is substantial enough, add it
            if current_tokens >= min_tokens:
                # Create metadata for this chunk
                chunk_metadata = metadata.copy()
                citation = extract_citation(current_chunk)
                if citation:
                    chunk_metadata["citation"] = citation

                chunks.append(Chunk(
                    text=current_chunk,
                    metadata=chunk_metadata,
                    tokens=current_tokens
                ))

            # Start a new chunk with this paragraph
            current_chunk = paragraph
            current_tokens = paragraph_tokens

    # Add the last chunk if it exists
    if current_chunk and current_tokens >= min_tokens:
        # Create metadata for this chunk
        chunk_metadata = metadata.copy()
        citation = extract_citation(current_chunk)
        if citation:
            chunk_metadata["citation"] = citation

        chunks.append(Chunk(
            text=current_chunk,
            metadata=chunk_metadata,
            tokens=current_tokens
        ))

    return chunks

def process_legal_document(document: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
    """Process a legal document using the appropriate chunking strategy."""
    if metadata is None:
        metadata = {}

    # Extract basic metadata if not provided
    if "document_type" not in metadata:
        metadata["document_type"] = identify_document_type(document)

    if "jurisdiction" not in metadata:
        jurisdiction = identify_jurisdiction(document)
        if jurisdiction:
            metadata["jurisdiction"] = jurisdiction

    if "citation" not in metadata:
        citation = extract_citation(document)
        if citation:
            metadata["citation"] = citation

    # Use strategies based on configuration
    chunking_strategies = config["chunking"]["strategies"]

    if chunking_strategies.get("case_document", {}).get("enabled", False):
        case_chunks = chunk_document_by_case(document, metadata)

        # If the document is small enough, just use case-level chunking
        if case_chunks and case_chunks[0].tokens <= CHUNK_SIZES.get("legal_reasoning", {}).get("max_tokens", 700):
            return case_chunks

    # Use section-level chunking
    if chunking_strategies.get("legal_sections", {}).get("enabled", False):
        section_chunks = chunk_document_by_sections(document, metadata)

        # If sections are appropriately sized, use section-level chunking
        if all(chunk.tokens <= CHUNK_SIZES.get("legal_reasoning", {}).get("max_tokens", 700) for chunk in section_chunks):
            return section_chunks

    # If we need more granular chunking, use paragraph-level
    if chunking_strategies.get("paragraphs", {}).get("enabled", False):
        section_chunks = chunk_document_by_sections(document, metadata)
        paragraph_chunks = []

        for chunk in section_chunks:
            # If this section chunk is too large, break it down by paragraphs
            if chunk.tokens > CHUNK_SIZES.get("legal_reasoning", {}).get("max_tokens", 700):
                min_tokens = CHUNK_SIZES.get(extract_section_type(chunk.text), {}).get("min_tokens", 400)
                max_tokens = CHUNK_SIZES.get(extract_section_type(chunk.text), {}).get("max_tokens", 600)
                paragraph_chunks.extend(chunk_by_paragraphs(chunk.text, chunk.metadata, min_tokens, max_tokens))
            else:
                paragraph_chunks.append(chunk)

        return paragraph_chunks

    # Fallback to basic paragraph chunking if no strategy is enabled
    return chunk_by_paragraphs(
        document,
        metadata,
        CHUNK_SIZES.get("legal_reasoning", {}).get("min_tokens", 400),
        CHUNK_SIZES.get("legal_reasoning", {}).get("max_tokens", 600)
    )