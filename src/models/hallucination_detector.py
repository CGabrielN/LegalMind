"""
LegalMind Hallucination Detection and Mitigation

This module implements detection and mitigation strategies for hallucinations
in legal responses, with a focus on verifying citations and claims against
the retrieved context.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LegalHallucinationDetector:
    """
    Detects and mitigates hallucinations in legal responses.
    """

    def __init__(self):
        """Initialize the hallucination detector."""
        # Load a sentence transformer model for semantic similarity
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight model for semantic similarity
        self.model = SentenceTransformer(self.model_name)

        # Confidence threshold for factual claims
        self.confidence_threshold = 0.7

        # Legal citation pattern (Australian format)
        self.citation_pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"

        # Definitive phrase patterns that should be verified
        self.definitive_patterns = [
            r"always requires",
            r"never allows",
            r"all cases",
            r"in every situation",
            r"without exception",
            r"settled law",
            r"established principle",
            r"landmark case",
            r"leading authority"
        ]

        logger.info("Initialized legal hallucination detector")

    def extract_citations(self, text: str) -> List[str]:
        """
        Extract legal citations from text.

        Args:
            text: Text to analyze

        Returns:
            List of extracted citations
        """
        citations = re.findall(self.citation_pattern, text)
        return citations

    def extract_definitive_claims(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract definitive legal claims that should be verified.

        Args:
            text: Text to analyze

        Returns:
            List of (claim_text, pattern_matched) tuples
        """
        claims = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            for pattern in self.definitive_patterns:
                match = re.search(pattern, sentence.lower())
                if match:
                    claims.append((sentence, pattern))
                    break

        return claims


    @staticmethod
    def verify_citation(citation: str, context: str) -> Tuple[bool, float]:
        """
        Verify if a citation exists in the context.

        Args:
            citation: Legal citation to verify
            context: Retrieved context

        Returns:
            Tuple of (is_verified, confidence_score)
        """
        try:
            # Strip any leading/trailing whitespace that could cause issues
            citation = citation.strip()

            # Direct string match (most reliable)
            if citation in context:
                return True, 1.0

            # If no direct match, look for parts of the citation
            parts = citation.split()

            # Make sure we have enough parts before trying to access them
            if len(parts) >= 4:
                try:
                    # Find the index of the bracket, which is safer than assuming position
                    bracket_index = -1
                    for i, part in enumerate(parts):
                        if part.startswith("["):
                            bracket_index = i
                            break

                    if bracket_index == -1:
                        # No bracket found, try a simple check
                        if any(part in context for part in parts):
                            return True, 0.3
                        return False, 0.0

                    # Extract case name and year safely
                    case_name = " ".join(parts[:bracket_index])
                    year = parts[bracket_index].strip("[]")

                    # Check if both case name and year appear nearby in the context
                    if case_name in context and year in context:
                        # Check if they appear within reasonable proximity
                        context_lower = context.lower()
                        case_pos = context_lower.find(case_name.lower())
                        year_pos = context_lower.find(year)

                        # If they're within 100 characters, consider it a valid reference
                        if abs(case_pos - year_pos) < 100:
                            return True, 0.8
                except Exception as e:
                    # If any exception occurs in the detailed parsing, use a fallback
                    # Look for any major parts in the context
                    score = 0.0
                    for part in parts:
                        if len(part) > 3 and part in context:
                            score += 0.1
                    return score > 0.3, min(score, 0.6)

            return False, 0.0
        except Exception as e:
            # Ultimate fallback - just check if any substantial part appears in the context
            significant_parts = [p for p in citation.split() if len(p) > 3]
            for part in significant_parts:
                if part in context:
                    return True, 0.3
            return False, 0.0

    def verify_claim(self, claim: str, context: str) -> Tuple[bool, float]:
        """
        Verify if a legal claim is supported by the context.

        Args:
            claim: Legal claim to verify
            context: Retrieved context

        Returns:
            Tuple of (is_verified, confidence_score)
        """
        # Embed the claim and context paragraphs
        context_paragraphs = context.split('\n\n')

        # If there are too many paragraphs, combine some to avoid too many comparisons
        if len(context_paragraphs) > 10:
            combined_paragraphs = []
            for i in range(0, len(context_paragraphs), 2):
                if i+1 < len(context_paragraphs):
                    combined_paragraphs.append(context_paragraphs[i] + " " + context_paragraphs[i+1])
                else:
                    combined_paragraphs.append(context_paragraphs[i])
            context_paragraphs = combined_paragraphs

        # Filter out very short paragraphs
        context_paragraphs = [p for p in context_paragraphs if len(p.split()) > 10]

        # If no substantial paragraphs, return not verified
        if not context_paragraphs:
            return False, 0.0

        # Embed claim and paragraphs
        claim_embedding = self.model.encode([claim])[0]
        paragraph_embeddings = self.model.encode(context_paragraphs)

        # Calculate similarities
        similarities = cosine_similarity([claim_embedding], paragraph_embeddings)[0]

        # Get max similarity
        max_sim = np.max(similarities)

        # Determine if claim is verified based on similarity threshold
        is_verified = max_sim >= self.confidence_threshold

        return is_verified, float(max_sim)

    def check_for_hallucinations(self, response: str, context: str) -> Dict[str, Any]:
        """
        Check for hallucinations in a legal response.

        Args:
            response: Generated legal response
            context: Retrieved context used for generation

        Returns:
            Dictionary with hallucination analysis
        """
        logger.info("Checking for hallucinations in response")

        # Initialize results
        results = {
            "has_hallucinations": False,
            "citation_checks": [],
            "claim_checks": [],
            "overall_confidence": 1.0
        }

        try:
            # 1. Check citations
            citations = self.extract_citations(response)
            citation_confidence = []

            for citation in citations:
                try:
                    verified, confidence = self.verify_citation(citation, context)
                    citation_checks = {
                        "citation": citation,
                        "verified": verified,
                        "confidence": confidence
                    }
                    results["citation_checks"].append(citation_checks)
                    citation_confidence.append(confidence)

                    if not verified:
                        results["has_hallucinations"] = True
                except Exception as e:
                    logger.warning(f"Error verifying citation '{citation}': {str(e)}")
                    # Add a failed check with low confidence
                    results["citation_checks"].append({
                        "citation": citation,
                        "verified": False,
                        "confidence": 0.1
                    })
                    citation_confidence.append(0.1)
                    results["has_hallucinations"] = True

            # 2. Check definitive claims
            claims = self.extract_definitive_claims(response)
            claim_confidence = []

            for claim_text, pattern_matched in claims:
                try:
                    verified, confidence = self.verify_claim(claim_text, context)
                    claim_check = {
                        "claim": claim_text,
                        "pattern": pattern_matched,
                        "verified": verified,
                        "confidence": confidence
                    }
                    results["claim_checks"].append(claim_check)
                    claim_confidence.append(confidence)

                    if not verified:
                        results["has_hallucinations"] = True
                except Exception as e:
                    logger.warning(f"Error verifying claim '{claim_text}': {str(e)}")
                    results["claim_checks"].append({
                        "claim": claim_text,
                        "pattern": pattern_matched,
                        "verified": False,
                        "confidence": 0.1
                    })
                    claim_confidence.append(0.1)
                    results["has_hallucinations"] = True

            # Calculate overall confidence
            if citation_confidence or claim_confidence:
                all_confidence = citation_confidence + claim_confidence
                results["overall_confidence"] = sum(all_confidence) / len(all_confidence)

        except Exception as e:
            logger.error(f"Error in hallucination check: {str(e)}")
            # Return sensible defaults if the process fails
            results["has_hallucinations"] = False
            results["overall_confidence"] = 0.7

        # Log results
        logger.info(f"Hallucination check complete. Has hallucinations: {results['has_hallucinations']}")
        logger.info(f"Checked {len(citations)} citations and {len(claims)} definitive claims")

        return results

    def mitigate_hallucinations(self, response: str, context: str, analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Mitigate hallucinations by modifying the response.

        Args:
            response: Original response
            context: Retrieved context
            analysis: Optional pre-computed hallucination analysis

        Returns:
            Modified response with hallucinations mitigated
        """
        # If analysis not provided, compute it
        if analysis is None:
            analysis = self.check_for_hallucinations(response, context)

        # If no hallucinations, return original response
        if not analysis["has_hallucinations"]:
            return response

        logger.info("Mitigating hallucinations in response")

        # Create modified response
        modified_response = response

        # 1. Handle unverified citations
        for check in analysis["citation_checks"]:
            if not check["verified"]:
                citation = check["citation"]

                # Try to find a similar citation in the context
                context_citations = self.extract_citations(context)

                # If we found similar citations, use one of those instead
                if context_citations:
                    # Try to find one with similar case name
                    citation_parts = citation.split()
                    if len(citation_parts) >= 2:
                        case_name_part = citation_parts[0]

                        for context_citation in context_citations:
                            if case_name_part in context_citation:
                                # Replace the unverified citation with a verified one
                                modified_response = modified_response.replace(citation, context_citation)
                                logger.info(f"Replaced unverified citation {citation} with {context_citation}")
                                break

                # If no replacement found, add a disclaimer
                if citation in modified_response:
                    modified_response = modified_response.replace(
                        citation,
                        f"{citation} (Note: This citation should be verified)"
                    )
                    logger.info(f"Added disclaimer to unverified citation: {citation}")

        # 2. Handle unverified claims
        for check in analysis["claim_checks"]:
            if not check["verified"]:
                claim = check["claim"]

                # Replace definitive language with more cautious wording
                for pattern in self.definitive_patterns:
                    if re.search(pattern, claim.lower()):
                        # Replace with more cautious language
                        new_claim = re.sub(
                            pattern,
                            self._get_cautious_replacement(pattern),
                            claim,
                            flags=re.IGNORECASE
                        )

                        modified_response = modified_response.replace(claim, new_claim)
                        logger.info(f"Modified definitive claim: {claim} → {new_claim}")

        # 3. Add general disclaimer if significant modifications were made
        if modified_response != response:
            disclaimer = "\n\nNote: Some statements in this response are based on general legal principles and may not reflect the most current precedents. Please verify citations and consult a legal professional for advice specific to your situation."

            # Add disclaimer if not already present
            if disclaimer not in modified_response:
                modified_response += disclaimer

        return modified_response

    @staticmethod
    def _get_cautious_replacement(pattern: str) -> str:
        """Get a cautious replacement for a definitive pattern."""
        replacements = {
            "always requires": "often requires",
            "never allows": "rarely allows",
            "all cases": "many cases",
            "in every situation": "in many situations",
            "without exception": "with few exceptions",
            "settled law": "generally accepted legal principle",
            "established principle": "common legal principle",
            "landmark case": "significant case",
            "leading authority": "important authority"
        }

        return replacements.get(pattern, "may")

class CitationVerifier:
    """
    Specialized verification of Australian legal citations.
    """

    def __init__(self):
        """Initialize the citation verifier."""
        # Load jurisdiction abbreviations
        self.jurisdictions = {
            "HCA": "High Court of Australia",
            "FCAFC": "Federal Court of Australia (Full Court)",
            "FCA": "Federal Court of Australia",
            "NSWSC": "New South Wales Supreme Court",
            "NSWCA": "New South Wales Court of Appeal",
            "VSC": "Victoria Supreme Court",
            "VSCA": "Victoria Court of Appeal",
            "QSC": "Queensland Supreme Court",
            "QCA": "Queensland Court of Appeal",
            "WASC": "Western Australia Supreme Court",
            "WASCA": "Western Australia Court of Appeal",
            "SASC": "South Australia Supreme Court",
            "SASCFC": "South Australia Supreme Court (Full Court)",
            "TASSC": "Tasmania Supreme Court",
            "NTSC": "Northern Territory Supreme Court",
            "ACTSC": "Australian Capital Territory Supreme Court"
        }

        logger.info("Initialized citation verifier")

    def parse_citation(self, citation: str) -> Dict[str, Any]:
        """
        Parse an Australian legal citation.

        Args:
            citation: Citation string

        Returns:
            Dictionary with parsed components
        """
        # Pattern: Case name v Respondent [Year] Court Number
        # Example: "Smith v Jones [2010] NSWSC 123"

        parts = {}

        try:
            # Extract case name
            name_match = re.match(r"(.+?)\s+\[", citation)
            if name_match:
                parts["case_name"] = name_match.group(1).strip()

            # Extract year
            year_match = re.search(r"\[(\d{4})\]", citation)
            if year_match:
                parts["year"] = year_match.group(1)

            # Extract court and number
            court_match = re.search(r"\[\d{4}\]\s+([A-Z]+)\s+(\d+)", citation)
            if court_match:
                court_abbr = court_match.group(1)
                parts["court_abbr"] = court_abbr
                parts["court"] = self.jurisdictions.get(court_abbr, court_abbr)
                parts["number"] = court_match.group(2)

            return parts

        except Exception as e:
            logger.warning(f"Error parsing citation '{citation}': {str(e)}")
            return {"original": citation}

    @staticmethod
    def validate_citation_format(citation: str) -> bool:
        """
        Validate if a citation follows the expected Australian format.

        Args:
            citation: Citation string

        Returns:
            Whether the citation is valid
        """
        # Basic pattern for Australian citations
        pattern = r"^[A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+$"
        return bool(re.match(pattern, citation))

    def check_jurisdiction_consistency(self, response: str, context: str) -> Dict[str, Any]:
        """
        Check if response is consistent with jurisdictions in context.

        Args:
            response: Generated response
            context: Retrieved context

        Returns:
            Dictionary with consistency analysis
        """
        # Initialize results with default values in case of empty context
        results = {
            "jurisdiction_match": True,
            "response_jurisdictions": {},
            "context_jurisdictions": {},
            "mismatched_jurisdictions": [],
            "confidence_score": 1.0
        }

        try:
            # Identify jurisdictions in response and context
            response_jurs = self.identify_jurisdictions(response)

            # If context is empty, assume all jurisdictions are valid
            if not context.strip():
                logger.warning("Empty context provided for jurisdiction consistency check")
                return results

            context_jurs = self.identify_jurisdictions(context)

            # Update results
            results["response_jurisdictions"] = response_jurs
            results["context_jurisdictions"] = context_jurs

            # Check for jurisdictions in response not supported by context
            for jur, score in response_jurs.items():
                if jur not in context_jurs or context_jurs[jur] < score / 2:
                    results["jurisdiction_match"] = False
                    results["mismatched_jurisdictions"].append(jur)

            # Calculate confidence score
            if results["mismatched_jurisdictions"]:
                # Reduce confidence proportionally to mismatch
                mismatch_scores = [response_jurs[jur] for jur in results["mismatched_jurisdictions"]]
                results["confidence_score"] = max(0.0, 1.0 - sum(mismatch_scores) / len(mismatch_scores))

        except Exception as e:
            logger.error(f"Error in jurisdiction consistency check: {str(e)}")
            # Return default values if there's an error

        return results

    def find_similar_citations(self, citation: str, context: str) -> List[str]:
        """
        Find similar citations in the context.

        Args:
            citation: Citation string
            context: Retrieved context

        Returns:
            List of similar citations found in context
        """
        # Parse the citation to extract components
        parts = self.parse_citation(citation)

        # If parsing failed, return empty list
        if "case_name" not in parts:
            return []

        # Extract all citations from context
        all_citations = re.findall(r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)", context)

        # Filter for citations with similar case name
        similar_citations = []

        case_name = parts.get("case_name", "")
        case_parties = case_name.split(" v ")

        for ctx_citation in all_citations:
            # Parse context citation
            ctx_parts = self.parse_citation(ctx_citation)

            # If parsing failed, skip
            if "case_name" not in ctx_parts:
                continue

            # Check for similar case name
            ctx_case_name = ctx_parts["case_name"]
            ctx_parties = ctx_case_name.split(" v ")

            # If at least one party name matches
            if len(case_parties) > 0 and len(ctx_parties) > 0:
                if case_parties[0].lower() in ctx_parties[0].lower() or \
                        (len(case_parties) > 1 and len(ctx_parties) > 1 and
                         case_parties[1].lower() in ctx_parties[1].lower()):
                    similar_citations.append(ctx_citation)

            # Also check for same year and court
            elif parts.get("year") == ctx_parts.get("year") and \
                    parts.get("court_abbr") == ctx_parts.get("court_abbr"):
                similar_citations.append(ctx_citation)

        return similar_citations

class ConfidenceScoring:
    """
    Scores confidence in legal responses for jurisdiction-specific claims.
    """

    def __init__(self):
        """Initialize the confidence scorer."""
        # Keywords that indicate jurisdiction-specific content
        self.jurisdiction_keywords = {
            "nsw": ["new south wales", "nsw", "sydney"],
            "vic": ["victoria", "vic", "melbourne"],
            "qld": ["queensland", "qld", "brisbane"],
            "wa": ["western australia", "wa", "perth"],
            "sa": ["south australia", "sa", "adelaide"],
            "tas": ["tasmania", "tas", "hobart"],
            "nt": ["northern territory", "nt", "darwin"],
            "act": ["australian capital territory", "act", "canberra"],
            "federal": ["federal", "commonwealth", "australia", "high court"]
        }

        # Legal area keywords
        self.legal_areas = {
            "contract": ["contract", "agreement", "offer", "acceptance", "consideration"],
            "tort": ["tort", "negligence", "duty of care", "damages", "liability"],
            "property": ["property", "land", "title", "possession", "easement"],
            "criminal": ["criminal", "crime", "offense", "guilty", "prosecution"],
            "family": ["family", "divorce", "custody", "marriage", "child support"],
            "employment": ["employment", "worker", "workplace", "unfair dismissal"],
            "consumer": ["consumer", "goods", "services", "warranty", "refund"]
        }

        # Initialize embeddings model for semantic similarity
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        logger.info("Initialized confidence scoring system")

    def identify_jurisdictions(self, text: str) -> Dict[str, float]:
        """
        Identify jurisdictions mentioned in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of jurisdictions and their confidence scores
        """
        text_lower = text.lower()
        scores = {}

        # Check for mentions of each jurisdiction
        for jur, keywords in self.jurisdiction_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                scores[jur] = min(1.0, count / 3)  # Cap at 1.0

        return scores

    def identify_legal_areas(self, text: str) -> Dict[str, float]:
        """
        Identify legal areas mentioned in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of legal areas and their confidence scores
        """
        text_lower = text.lower()
        scores = {}

        # Check for mentions of each legal area
        for area, keywords in self.legal_areas.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                scores[area] = min(1.0, count / 3)  # Cap at 1.0

        return scores

    def check_jurisdiction_consistency(self, response: str, context: str) -> Dict[str, Any]:
        """
        Check if response is consistent with jurisdictions in context.

        Args:
            response: Generated response
            context: Retrieved context

        Returns:
            Dictionary with consistency analysis
        """
        # Identify jurisdictions in response and context
        response_jurs = self.identify_jurisdictions(response)
        context_jurs = self.identify_jurisdictions(context)

        # Initialize results
        results = {
            "jurisdiction_match": True,
            "response_jurisdictions": response_jurs,
            "context_jurisdictions": context_jurs,
            "mismatched_jurisdictions": [],
            "confidence_score": 1.0
        }

        # Check for jurisdictions in response not supported by context
        for jur, score in response_jurs.items():
            if jur not in context_jurs or context_jurs[jur] < score / 2:
                results["jurisdiction_match"] = False
                results["mismatched_jurisdictions"].append(jur)

        # Calculate confidence score
        if results["mismatched_jurisdictions"]:
            # Reduce confidence proportionally to mismatch
            mismatch_scores = [response_jurs[jur] for jur in results["mismatched_jurisdictions"]]
            results["confidence_score"] = max(0.0, 1.0 - sum(mismatch_scores) / len(mismatch_scores))

        return results

    def calculate_response_confidence(self, response: str, context: str, citation_checks: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence in a legal response.

        Args:
            response: Generated response
            context: Retrieved context
            citation_checks: Results of citation verification

        Returns:
            Confidence score (0.0-1.0)
        """
        # Check jurisdiction consistency
        jurisdiction_results = self.check_jurisdiction_consistency(response, context)
        jurisdiction_score = jurisdiction_results["confidence_score"]

        # Calculate citation confidence
        citation_scores = [check["confidence"] for check in citation_checks]
        citation_score = sum(citation_scores) / len(citation_scores) if citation_scores else 1.0

        # Calculate semantic similarity between response and context
        # But only if context is not empty
        if not context.strip():
            # If context is empty, we can't calculate semantic similarity
            semantic_score = 0.5  # Use a neutral score
            logger.warning("Empty context provided for semantic similarity calculation")
        else:
            try:
                response_embedding = self.model.encode([response])[0]

                # Split context into chunks to handle large contexts
                context_chunks = [context[i:i+1000] for i in range(0, len(context), 1000)]

                # Only proceed if we have context chunks
                if not context_chunks:
                    semantic_score = 0.5
                else:
                    chunk_embeddings = self.model.encode(context_chunks)

                    # Calculate max similarity with any chunk
                    similarities = cosine_similarity([response_embedding], chunk_embeddings)[0]
                    semantic_score = float(np.max(similarities))
            except Exception as e:
                logger.error(f"Error calculating semantic similarity: {str(e)}")
                semantic_score = 0.5  # Use a neutral score if calculation fails

        # Combine scores with weights
        weights = {
            "jurisdiction": 0.3,
            "citation": 0.4,
            "semantic": 0.3
        }

        combined_score = (
                weights["jurisdiction"] * jurisdiction_score +
                weights["citation"] * citation_score +
                weights["semantic"] * semantic_score
        )

        logger.info(f"Confidence scores - Jurisdiction: {jurisdiction_score:.2f}, Citation: {citation_score:.2f}, Semantic: {semantic_score:.2f}")
        logger.info(f"Overall confidence: {combined_score:.2f}")

        return combined_score

class HallucinationMitigation:
    """
    Comprehensive hallucination detection and mitigation for legal responses.
    """

    def __init__(self):
        """Initialize hallucination mitigation system."""
        self.detector = LegalHallucinationDetector()
        self.citation_verifier = CitationVerifier()
        self.confidence_scorer = ConfidenceScoring()

        logger.info("Initialized hallucination mitigation system")

    def analyze_response(self, response: str, context: str) -> Dict[str, Any]:
        """
        Analyze a response for potential hallucinations.

        Args:
            response: Generated response
            context: Retrieved context

        Returns:
            Comprehensive analysis results
        """
        logger.info("Performing comprehensive hallucination analysis")

        # Basic hallucination detection
        hallucination_results = self.detector.check_for_hallucinations(response, context)

        # Extract and validate citations
        citations = self.detector.extract_citations(response)
        citation_checks = []

        for citation in citations:
            # Parse citation
            parsed = self.citation_verifier.parse_citation(citation)

            # Check if citation is in context
            verified, confidence = self.detector.verify_citation(citation, context)

            # Find similar citations in context
            similar_citations = self.citation_verifier.find_similar_citations(citation, context)

            citation_checks.append({
                "citation": citation,
                "parsed": parsed,
                "verified": verified,
                "confidence": confidence,
                "similar_citations": similar_citations
            })

        # Calculate confidence score
        confidence_score = self.confidence_scorer.calculate_response_confidence(
            response, context, citation_checks
        )

        # Perform jurisdiction analysis
        jurisdiction_results = self.confidence_scorer.check_jurisdiction_consistency(response, context)

        # Compile results
        results = {
            "has_hallucinations": hallucination_results["has_hallucinations"],
            "citation_checks": citation_checks,
            "jurisdiction_analysis": jurisdiction_results,
            "confidence_score": confidence_score,
            "claim_checks": hallucination_results["claim_checks"],
            "hallucination_severity": self._calculate_severity(hallucination_results, jurisdiction_results)
        }

        logger.info(f"Analysis complete - Hallucination severity: {results['hallucination_severity']}")
        return results

    @staticmethod
    def _calculate_severity(hallucination_results: Dict[str, Any], jurisdiction_results: Dict[str, Any]) -> str:
        """Calculate the severity of hallucinations."""
        # Count unverified citations and claims
        unverified_citations = sum(1 for check in hallucination_results["citation_checks"] if not check["verified"])
        unverified_claims = sum(1 for check in hallucination_results["claim_checks"] if not check["verified"])

        # Check jurisdiction mismatches
        jurisdiction_mismatches = len(jurisdiction_results["mismatched_jurisdictions"])

        # Determine severity
        if unverified_citations > 2 or jurisdiction_mismatches > 0:
            return "high"
        elif unverified_citations > 0 or unverified_claims > 1:
            return "medium"
        elif unverified_claims > 0:
            return "low"
        else:
            return "none"

    def mitigate_hallucinations(self, response: str, context: str, analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Mitigate hallucinations in a response.

        Args:
            response: Original response
            context: Retrieved context
            analysis: Optional pre-computed analysis

        Returns:
            Mitigated response
        """
        # If analysis not provided, perform it
        if analysis is None:
            analysis = self.analyze_response(response, context)

        # If no hallucinations, return original response
        if analysis["hallucination_severity"] == "none":
            return response

        logger.info(f"Mitigating hallucinations with severity: {analysis['hallucination_severity']}")

        # Create modified response
        modified_response = response

        # 1. Fix unverified citations
        for check in analysis["citation_checks"]:
            if not check["verified"]:
                citation = check["citation"]
                similar_citations = check["similar_citations"]

                if similar_citations:
                    # Replace with the first similar citation
                    modified_response = modified_response.replace(citation, similar_citations[0])
                    logger.info(f"Replaced citation: {citation} → {similar_citations[0]}")
                else:
                    # Add a disclaimer
                    modified_response = modified_response.replace(
                        citation,
                        f"{citation} [citation may not be accurate]"
                    )
                    logger.info(f"Added disclaimer to citation: {citation}")

        # 2. Address jurisdiction inconsistencies
        if not analysis["jurisdiction_analysis"]["jurisdiction_match"]:
            # Add jurisdiction disclaimer
            mismatched = ", ".join(analysis["jurisdiction_analysis"]["mismatched_jurisdictions"]).upper()
            disclaimer = f"\n\nNote: The information provided about {mismatched} jurisdiction may not be based on the most relevant or current authorities. Please consult specific legal resources for this jurisdiction."

            if disclaimer not in modified_response:
                modified_response += disclaimer
                logger.info(f"Added jurisdiction disclaimer for: {mismatched}")

        # 3. Add confidence level based on severity
        severity = analysis["hallucination_severity"]
        confidence = analysis["confidence_score"]

        if severity == "high":
            confidence_note = "\n\nNote: This response should be treated as a general overview only and may contain statements not fully supported by Australian legal authorities. Please verify all information with appropriate legal resources before relying on it."
        elif severity == "medium":
            confidence_note = "\n\nNote: While this response is based on Australian legal principles, some specific statements may benefit from additional verification."
        elif severity == "low":
            confidence_note = "\n\nNote: This response is generally reliable but, as with all legal information, should be verified with appropriate legal resources."
        else:
            confidence_note = ""

        if confidence_note and confidence_note not in modified_response:
            modified_response += confidence_note

        logger.info("Completed hallucination mitigation")
        return modified_response