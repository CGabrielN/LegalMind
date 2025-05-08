"""
LegalMind Hallucination Detection and Mitigation Pipeline

This module integrates the RLHF, citation handling, and hallucination detection
components into a unified pipeline for generating high-quality legal responses.
"""

import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    from .enhanced_rlhf import EnhancedLegalRLHF as LegalRLHF
    USING_ENHANCED_RLHF = True
except ImportError:
    from .rlhf import LegalRLHF
    USING_ENHANCED_RLHF = False

from .hallucination_detector import HallucinationMitigation
from .citation_handler import CitationCrossReferencer, CitationVerification, CitationFormatting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class LegalResponsePipeline:
    """
    Unified pipeline for generating, evaluating, and improving legal responses.
    """

    def __init__(self):
        """Initialize the response pipeline with all components."""
        # Initialize RLHF system if enabled
        self.rlhf_enabled = config["rlhf"]["enabled"]
        if self.rlhf_enabled:
            self.rlhf = LegalRLHF()
        else:
            self.rlhf = None

        # Initialize hallucination mitigation
        self.hallucination_enabled = config["hallucination"]["enabled"]
        if self.hallucination_enabled:
            self.hallucination_mitigation = HallucinationMitigation()
        else:
            self.hallucination_mitigation = None

        # Initialize citation components
        self.citation_verification_enabled = config["citation"]["verification"]["enabled"]
        if self.citation_verification_enabled:
            self.citation_verifier = CitationVerification()
        else:
            self.citation_verifier = None

        self.citation_formatting_enabled = config["citation"]["formatting"]["enabled"]
        if self.citation_formatting_enabled:
            self.citation_formatter = CitationFormatting()
        else:
            self.citation_formatter = None

        # Cross-referencer for citations
        self.citation_crosschecker = CitationCrossReferencer()

        # Response confidence threshold
        self.confidence_threshold = config["rlhf"]["confidence_threshold"]

        logger.info("Initialized legal response pipeline")
        logger.info(f"RLHF enabled: {self.rlhf_enabled}")
        logger.info(f"Hallucination mitigation enabled: {self.hallucination_enabled}")
        logger.info(f"Citation verification enabled: {self.citation_verification_enabled}")
        logger.info(f"Citation formatting enabled: {self.citation_formatting_enabled}")

    def generate_improved_response(self,
                                   query: str,
                                   context: str,
                                   response: str) -> Dict[str, Any]:
        """
        Process a generated response through the pipeline for improvement.

        Args:
            query: The legal query
            context: Retrieved context provided to LLM
            response: Initial LLM response

        Returns:
            Dictionary with improved response and analysis
        """
        logger.info(f"Processing response for query: '{query[:50]}...'")

        # Initialize result dictionary
        result = {
            "original_response": response,
            "improved_response": response,
            "confidence_score": 1.0,
            "has_hallucinations": False,
            "citation_analysis": {},
            "hallucination_analysis": {},
            "rlhf_score": None
        }

        # Step 1: Verify citations
        if self.citation_verification_enabled and self.citation_verifier:
            logger.info("Verifying citations in response")
            citation_analysis = self.citation_verifier.verify_all_citations_in_response(response, context)
            result["citation_analysis"] = citation_analysis
            result["has_hallucinations"] = citation_analysis.get("has_unverified_citations", False)

        # Step 2: Detect hallucinations
        if self.hallucination_enabled and self.hallucination_mitigation:
            logger.info("Analyzing response for hallucinations")
            hallucination_analysis = self.hallucination_mitigation.analyze_response(response, context, query)
            result["hallucination_analysis"] = hallucination_analysis
            result["has_hallucinations"] = result["has_hallucinations"] or hallucination_analysis.get("has_hallucinations", False)
            result["confidence_score"] = hallucination_analysis.get("confidence_score", 1.0)

        # Step 3: Evaluate with RLHF model if available
        if self.rlhf_enabled and self.rlhf:
            logger.info("Evaluating response with RLHF reward model")
            rlhf_score = self.rlhf.evaluate_response(query, response)
            result["rlhf_score"] = rlhf_score

            # Adjust confidence score based on RLHF evaluation
            if rlhf_score is not None:
                result["confidence_score"] = (result["confidence_score"] + rlhf_score) / 2

        # Step 4: Mitigate hallucinations if necessary
        if result["has_hallucinations"] and self.hallucination_enabled and self.hallucination_mitigation:
            logger.info("Mitigating hallucinations in response")
            improved_response = self.hallucination_mitigation.mitigate_hallucinations(
                response,
                context,
                result["hallucination_analysis"]
            )
            result["improved_response"] = improved_response

        # Step 5: Format citations if enabled
        if self.citation_formatting_enabled and self.citation_formatter:
            logger.info("Formatting citations in response")
            formatted_response = self.citation_formatter.format_all_citations(result["improved_response"])
            result["improved_response"] = formatted_response

        # Log result summary
        if result["improved_response"] != response:
            logger.info("Response was improved by the pipeline")
        else:
            logger.info("No improvements made to original response")

        logger.info(f"Final confidence score: {result['confidence_score']:.2f}")
        return result

    def suggest_improvements(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """
        Suggest potential improvements to a legal response without making changes.

        Args:
            query: The legal query
            context: Retrieved context
            response: Generated response

        Returns:
            Dictionary with suggested improvements
        """
        logger.info(f"Analyzing response for potential improvements: '{query[:50]}...'")

        suggestions = {
            "citation_suggestions": [],
            "factual_corrections": [],
            "formatting_suggestions": [],
            "confidence_score": 1.0,
            "overall_quality": "high"
        }

        # Check for citation issues
        if self.citation_verification_enabled and self.citation_verifier:
            citation_analysis = self.citation_verifier.verify_all_citations_in_response(response, context)

            # Suggest corrections for unverified citations
            for result in citation_analysis["verification_results"]:
                if not result["verified_in_context"]:
                    suggestion = {
                        "citation": result["citation"],
                        "issue": "Citation not found in context",
                        "confidence": result["context_confidence"]
                    }

                    # Try to find similar citations
                    if self.citation_crosschecker:
                        similar = self.citation_crosschecker.extract_and_cross_reference(context)
                        suggestion["alternatives"] = [cite for cite in similar.get("citations", [])[:3]]

                    suggestions["citation_suggestions"].append(suggestion)

            # Check if additional citations could be added
            if self.citation_verifier:
                suggested_citations = self.citation_verifier.suggest_citations(response, context)

                for cite_suggestion in suggested_citations:
                    if cite_suggestion["citation"] not in response:
                        suggestions["citation_suggestions"].append({
                            "suggestion_type": "additional_citation",
                            "citation": cite_suggestion["citation"],
                            "relevance": cite_suggestion["relevance"],
                            "context_snippet": cite_suggestion["context_snippet"][:100] + "..."
                        })

        # Check for hallucinations
        if self.hallucination_enabled and self.hallucination_mitigation:
            hallucination_analysis = self.hallucination_mitigation.analyze_response(response, context, query)

            # Add factual corrections for hallucinations
            if hallucination_analysis.get("has_hallucinations", False):
                # Extract claims with issues
                for claim_check in hallucination_analysis.get("claim_checks", []):
                    if not claim_check.get("verified", True):
                        suggestions["factual_corrections"].append({
                            "claim": claim_check["claim"],
                            "issue": f"Claim contains definitive language ({claim_check['pattern']}) without support in context",
                            "suggestion": "Use more cautious language or provide citation"
                        })

                # Extract jurisdiction mismatches
                jurisdiction_analysis = hallucination_analysis.get("jurisdiction_analysis", {})
                if not jurisdiction_analysis.get("jurisdiction_match", True):
                    mismatched = jurisdiction_analysis.get("mismatched_jurisdictions", [])
                    suggestions["factual_corrections"].append({
                        "issue": f"Response discusses jurisdiction(s) not supported by context: {', '.join(mismatched)}",
                        "suggestion": "Limit discussion to jurisdictions present in context"
                    })

            # Update confidence score
            suggestions["confidence_score"] = hallucination_analysis.get("confidence_score", 1.0)

        # Check formatting
        if self.citation_formatting_enabled and self.citation_formatter:
            formatted = self.citation_formatter.format_all_citations(response)
            if formatted != response:
                suggestions["formatting_suggestions"].append({
                    "issue": "Inconsistent citation formatting",
                    "suggestion": "Standardize citation format according to Australian legal style"
                })

        # Determine overall quality based on findings
        if len(suggestions["factual_corrections"]) > 2 or len(suggestions["citation_suggestions"]) > 3:
            suggestions["overall_quality"] = "low"
        elif len(suggestions["factual_corrections"]) > 0 or len(suggestions["citation_suggestions"]) > 1:
            suggestions["overall_quality"] = "medium"

        logger.info(f"Completed improvement analysis. Quality: {suggestions['overall_quality']}")
        return suggestions

    def collect_feedback(self, query: str, responses: List[str], chosen_idx: int, feedback_text: Optional[str] = None):
        """
        Collect human feedback on multiple response options.

        Args:
            query: The legal query
            responses: List of generated responses
            chosen_idx: Index of the chosen (preferred) response
            feedback_text: Optional feedback text explaining preference
        """
        if not self.rlhf_enabled or not self.rlhf:
            logger.warning("RLHF is not enabled, feedback not collected")
            return

        logger.info(f"Collecting feedback for query: '{query[:50]}...'")

        # Pass feedback to RLHF component
        self.rlhf.collect_feedback(query, responses, chosen_idx, feedback_text)

        logger.info("Feedback collected and stored")

    def train_from_feedback(self):
        """Train the RLHF model from collected feedback."""
        if not self.rlhf_enabled or not self.rlhf:
            logger.warning("RLHF is not enabled, cannot train from feedback")
            return

        logger.info("Training RLHF model from collected feedback")

        # Train the reward model
        self.rlhf.train_from_feedback()

        logger.info("RLHF training completed")

    def initialize_pipeline(self, seed_preferences: bool = True):
        """
        Initialize the pipeline components with seed data if needed.

        Args:
            seed_preferences: Whether to seed RLHF with synthetic preferences
        """
        logger.info("Initializing legal response pipeline components")

        # Initialize RLHF with synthetic preferences if requested
        if self.rlhf_enabled and self.rlhf and seed_preferences:
            logger.info("Seeding RLHF with synthetic preferences")
            self.rlhf.initialize_with_synthetic_preferences()

        # Initialize citation crosschecker with basic Australian precedents
        from .citation_handler import initialize_australian_precedents
        precedents = initialize_australian_precedents()

        precedent_db_path = config["citation"]["precedent_db_path"]
        self.citation_crosschecker.precedents = precedents
        self.citation_crosschecker.save_precedents(precedent_db_path)

        logger.info("Pipeline initialization complete")

    def evaluate_response_quality(self,
                                  query: str,
                                  context: str,
                                  response: str,
                                  reference_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of response quality.

        Args:
            query: The legal query
            context: Retrieved context
            response: Generated response
            reference_answer: Optional reference answer for comparison

        Returns:
            Dictionary with quality evaluation metrics
        """
        logger.info(f"Evaluating response quality for query: '{query[:50]}...'")

        evaluation = {
            "citation_accuracy": 1.0,
            "hallucination_score": 1.0,
            "rlhf_score": None,
            "completeness": None,
            "overall_score": 1.0
        }

        # Check citation accuracy
        if self.citation_verification_enabled and self.citation_verifier:
            citation_analysis = self.citation_verifier.verify_all_citations_in_response(response, context)
            evaluation["citation_accuracy"] = citation_analysis.get("verification_rate", 1.0)

            # If verification rate is low, reduce overall score
            if evaluation["citation_accuracy"] < 0.7:
                logger.warning(f"Low citation accuracy: {evaluation['citation_accuracy']:.2f}")

        # Check for hallucinations
        if self.hallucination_enabled and self.hallucination_mitigation:
            hallucination_analysis = self.hallucination_mitigation.analyze_response(response, context, query)

            # Convert hallucination analysis to a score (1.0 = no hallucinations)
            if hallucination_analysis.get("has_hallucinations", False):
                severity = hallucination_analysis.get("hallucination_severity", "low")
                if severity == "high":
                    evaluation["hallucination_score"] = 0.3
                elif severity == "medium":
                    evaluation["hallucination_score"] = 0.6
                elif severity == "low":
                    evaluation["hallucination_score"] = 0.8
            else:
                evaluation["hallucination_score"] = 1.0

        # RLHF evaluation
        if self.rlhf_enabled and self.rlhf:
            evaluation["rlhf_score"] = self.rlhf.evaluate_response(query, response)

        # Completeness (if reference answer provided)
        if reference_answer:
            from src.evaluation.metrics import ResponseEvaluator
            evaluator = ResponseEvaluator()
            evaluation["completeness"] = evaluator.evaluate_response_completeness(response, reference_answer)

        # Calculate overall score
        scores = [
            evaluation["citation_accuracy"],
            evaluation["hallucination_score"]
        ]

        if evaluation["rlhf_score"] is not None:
            scores.append(evaluation["rlhf_score"])

        if evaluation["completeness"] is not None:
            scores.append(evaluation["completeness"])

        evaluation["overall_score"] = sum(scores) / len(scores)

        logger.info(f"Evaluation complete. Overall score: {evaluation['overall_score']:.2f}")
        return evaluation

    def get_version_info(self) -> Dict[str, Any]:
        """Get version information about all components."""
        return {
            "rlhf_enabled": self.rlhf_enabled,
            "hallucination_mitigation_enabled": self.hallucination_enabled,
            "citation_verification_enabled": self.citation_verification_enabled,
            "citation_formatting_enabled": self.citation_formatting_enabled,
            "confidence_threshold": self.confidence_threshold,
            "pipeline_version": "0.1.0"
        }