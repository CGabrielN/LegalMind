"""
LegalMind Hallucination Detection and Mitigation Pipeline

This module integrates the RLHF, citation handling, and hallucination detection
components into a unified pipeline for generating high-quality legal responses.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LegalResponsePipeline:
    """
    Unified pipeline for generating, evaluating, and improving legal responses.
    """

    def __init__(self, resource_manager=None):
        """
        Initialize the response pipeline with all components.

        Args:
            resource_manager: Shared ResourceManager instance
        """
        # Import here to avoid circular imports
        if resource_manager is None:
            from src.core.resource_manager import ResourceManager
            resource_manager = ResourceManager()

        self.resource_manager = resource_manager
        self.config = resource_manager.config

        # Initialize RLHF system if enabled
        self.rlhf_enabled = self.config["rlhf"]["enabled"]
        if self.rlhf_enabled:
            try:
                from .enhanced_rlhf import EnhancedLegalRLHF as LegalRLHF
            except ImportError:
                from .rlhf import LegalRLHF

            self.rlhf = LegalRLHF(self.config)
        else:
            self.rlhf = None

        # Get hallucination mitigation from resource manager
        self.hallucination_enabled = self.config["hallucination"]["enabled"]
        if self.hallucination_enabled:
            self.hallucination_mitigation = resource_manager.hallucination_mitigation
        else:
            self.hallucination_mitigation = None

        # Get citation components from resource manager
        self.citation_verification_enabled = self.config["citation"]["verification"]["enabled"]
        if self.citation_verification_enabled:
            self.citation_verifier = resource_manager.citation_verifier
        else:
            self.citation_verifier = None

        self.citation_formatting_enabled = self.config["citation"]["formatting"]["enabled"]
        if self.citation_formatting_enabled:
            self.citation_formatter = resource_manager.citation_formatter
        else:
            self.citation_formatter = None


        # Response confidence threshold
        self.confidence_threshold = self.config["rlhf"]["confidence_threshold"]

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
            hallucination_analysis = self.hallucination_mitigation.analyze_response(response, context)
            result["hallucination_analysis"] = hallucination_analysis
            result["has_hallucinations"] = result["has_hallucinations"] or hallucination_analysis.get(
                "has_hallucinations", False)
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
            if hasattr(self.rlhf, "initialize_with_synthetic_preferences"):
                self.rlhf.initialize_with_synthetic_preferences()

        # Initialize citation crosschecker with basic Australian precedents
        # from .citation_handler import initialize_australian_precedents
        # precedents = initialize_australian_precedents()

        precedent_db_path = self.config["citation"]["precedent_db_path"]
        # self.citation_crosschecker.precedents = precedents
        self.citation_crosschecker.save_precedents(precedent_db_path)

        logger.info("Pipeline initialization complete")
