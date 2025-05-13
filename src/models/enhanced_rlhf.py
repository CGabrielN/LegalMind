"""
LegalMind Enhanced RLHF Implementation

This module enhances the RLHF capabilities with specific focus on legal accuracy training.
It builds on the base RLHF implementation with improved feedback collection and training.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LegalFeedbackItem:
    """Represents a single feedback item for legal responses."""

    def __init__(self, query: str, response: str, is_positive: bool, feedback_text: Optional[str] = None):
        """Initialize with query, response and rating."""
        self.query = query
        self.response = response
        self.is_positive = is_positive
        self.feedback_text = feedback_text
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query": self.query,
            "response": self.response,
            "is_positive": self.is_positive,
            "feedback_text": self.feedback_text,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LegalFeedbackItem':
        """Create from dictionary."""
        item = cls(
            query=data["query"],
            response=data["response"],
            is_positive=data["is_positive"],
            feedback_text=data.get("feedback_text")
        )
        item.timestamp = data.get("timestamp", datetime.now().isoformat())
        return item


class EnhancedLegalRewardModel:
    """Enhanced reward model for evaluating legal response quality with improved training."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize reward model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path or os.path.join("models", "legal_reward_model")

        if os.path.exists(self.model_path):
            logger.info(f"Loading reward model from {self.model_path}")
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self._initialize_new_model()
        else:
            logger.info("No existing model found. Initializing new model.")
            self._initialize_new_model()

        self.model.to(self.device)
        self.max_length = 512

    def _initialize_new_model(self):
        """Initialize a new reward model."""
        # Use DistilBERT for a lightweight but effective model
        base_model = "distilbert-base-uncased"
        logger.info(f"Initializing new reward model from {base_model}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=1
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    def save_model(self):
        """Save the reward model."""
        os.makedirs(self.model_path, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"Saved reward model to {self.model_path}")

    def score_response(self, query: str, response: str) -> float:
        """
        Score a response given a query.

        Args:
            query: The legal query
            response: The response to evaluate

        Returns:
            Reward score (higher is better)
        """
        inputs = self.tokenizer(
            f"Query: {query}\n\nResponse: {response}",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Don't compute gradients
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the prediction (scalar reward value)
        reward = outputs.logits.item()

        # Normalize to 0-1 range with sigmoid
        normalized_reward = 1 / (1 + np.exp(-reward))

        return normalized_reward

    def create_better_response(self, query: str, initial_response: str, context: str, llm_api) -> str:
        """
        Try to create a better response based on reward model scoring.

        Args:
            query: The user query
            initial_response: The initial LLM response
            context: The retrieved context
            llm_api: The LLM API object to generate alternatives

        Returns:
            The improved response or the original if no improvement
        """
        # Score the initial response
        initial_score = self.score_response(query, initial_response)

        # If already high quality, return as is
        if initial_score > 0.9:
            logger.info(f"Initial response already high quality (score: {initial_score:.2f})")
            return initial_response

        # Try to generate an improved response
        try:
            # Create a prompt instructing the model to improve the response
            improvement_prompt = f"""
                You provided this response to a legal query, but it could be improved:

                QUERY: {query}

                YOUR RESPONSE:
                {initial_response}

                Please provide an improved response that:
                1. Includes more accurate legal citations
                2. Is more precise in legal terminology
                3. Clearly distinguishes between jurisdictions
                4. Avoids definitive statements without support

                IMPROVED RESPONSE:
                """

            # Generate a new response using the improvement prompt and context
            logger.info("Attempting to generate improved response")
            improved_response = llm_api.generate(improvement_prompt, context)

            # Score the improved response
            improved_score = self.score_response(query, improved_response)

            logger.info(f"Initial score: {initial_score:.2f}, Improved score: {improved_score:.2f}")

            # Only use the improved response if it scores better
            if improved_score > initial_score:
                logger.info("Using improved response")
                return improved_response
            else:
                logger.info("Keeping original response as improvement not better")
                return initial_response

        except Exception as e:
            logger.error(f"Error generating improved response: {str(e)}")
            return initial_response

    def train(self, preference_dataset, output_dir=None):
        """
        Train the reward model on preference data.

        Args:
            preference_dataset: Training dataset
            output_dir: Directory to save model (defaults to self.model_path)
        """
        if output_dir is None:
            output_dir = self.model_path

        os.makedirs(output_dir, exist_ok=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            save_steps=50,
            remove_unused_columns=False,
        )


        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=preference_dataset,
            compute_loss=self.compute_loss
        )

        # Train model
        logger.info("Training reward model...")
        trainer.train()

        # Save model
        self.save_model()
        logger.info("Reward model training complete")

        return True

    # Define custom loss function for preference learning
    def compute_loss(self, model, inputs):
        """Compute loss for preference learning."""
        # Process chosen responses
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"].to(self.device),
            attention_mask=inputs["chosen_attention_mask"].to(self.device)
        )

        # Process rejected responses
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"].to(self.device),
            attention_mask=inputs["rejected_attention_mask"].to(self.device)
        )

        # Get logits
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits

        # Calculate preference loss (log sigmoid of the difference)
        loss = -torch.log(torch.sigmoid(chosen_logits - rejected_logits)).mean()

        return loss



class EnhancedLegalPreferenceDataset(Dataset):
    """Enhanced dataset for training a preference model on legal responses."""

    def __init__(self, preference_pairs: List[Dict[str, Any]]):
        """Initialize with preference pairs."""
        logger.info(f"Creating dataset with {len(preference_pairs)} preference pairs")
        self.preference_pairs = preference_pairs

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = 512

    def __len__(self):
        """Return dataset length."""
        return len(self.preference_pairs)

    def __getitem__(self, idx):
        """Get tokenized item."""
        item = self.preference_pairs[idx]

        # Get query, chosen response and rejected response
        query = item["query"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize chosen response
        chosen_tokens = self.tokenizer(
            f"Query: {query}\n\nResponse: {chosen}",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize rejected response
        rejected_tokens = self.tokenizer(
            f"Query: {query}\n\nResponse: {rejected}",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
        }


class EnhancedLegalFeedbackCollector:
    """Enhanced feedback collection for legal responses with like/dislike integration."""

    def __init__(self, feedback_path: Optional[str] = None):
        """Initialize feedback collector."""
        self.feedback_path = feedback_path or os.path.join("data", "feedback")
        os.makedirs(self.feedback_path, exist_ok=True)

        # File for raw feedback data
        self.raw_feedback_file = os.path.join(self.feedback_path, "raw_legal_feedback.json")

        # File for processed preference pairs
        self.preference_pairs_file = os.path.join(self.feedback_path, "legal_preference_pairs.json")

        # Load existing feedback data
        self.raw_feedback = self._load_raw_feedback()
        self.preference_pairs = self._load_preference_pairs()

        logger.info(
            f"Initialized enhanced feedback collector with {len(self.raw_feedback)} items and {len(self.preference_pairs)} preference pairs")

    def _load_raw_feedback(self) -> List[Dict[str, Any]]:
        """Load raw feedback items."""
        if os.path.exists(self.raw_feedback_file):
            try:
                with open(self.raw_feedback_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding {self.raw_feedback_file}")
                return []
        return []

    def _load_preference_pairs(self) -> List[Dict[str, Any]]:
        """Load preference pairs."""
        if os.path.exists(self.preference_pairs_file):
            try:
                with open(self.preference_pairs_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding {self.preference_pairs_file}")
                return []
        return []

    def _save_raw_feedback(self):
        """Save raw feedback items."""
        with open(self.raw_feedback_file, 'w') as f:
            json.dump(self.raw_feedback, f, indent=2)

    def _save_preference_pairs(self):
        """Save preference pairs."""
        with open(self.preference_pairs_file, 'w') as f:
            json.dump(self.preference_pairs, f, indent=2)

    def collect_like_dislike_feedback(self, query: str, response: str, is_positive: bool,
                                      feedback_text: Optional[str] = None):
        """
        Collect like/dislike feedback on a response.

        Args:
            query: The legal query
            response: The response being rated
            is_positive: True for like, False for dislike
            feedback_text: Optional text explaining the rating
        """
        # Create feedback item
        feedback_item = LegalFeedbackItem(
            query=query,
            response=response,
            is_positive=is_positive,
            feedback_text=feedback_text
        )

        # Add to raw feedback
        self.raw_feedback.append(feedback_item.to_dict())
        self._save_raw_feedback()

        # Create synthetic preference pair
        self._create_preference_pair_from_feedback(feedback_item)

        logger.info(f"Collected {'positive' if is_positive else 'negative'} feedback for response")

    def _create_preference_pair_from_feedback(self, feedback_item: LegalFeedbackItem):
        """
        Create a preference pair from like/dislike feedback.

        For positive feedback, the liked response is chosen.
        For negative feedback, we create a synthetic better response.
        """
        query = feedback_item.query
        response = feedback_item.response

        if feedback_item.is_positive:
            # For positive feedback, we need a synthetic "worse" response
            # This would ideally be generated, but for simplicity we'll use templates
            rejected_response = self._generate_synthetic_worse_response(query, response)

            preference_pair = {
                "query": query,
                "chosen": response,
                "rejected": rejected_response,
                "feedback": "Positive user feedback, synthetic rejected response"
            }

        else:
            # For negative feedback, we need a synthetic "better" response
            chosen_response = self._generate_synthetic_better_response(query, response)

            preference_pair = {
                "query": query,
                "chosen": chosen_response,
                "rejected": response,
                "feedback": "Negative user feedback, synthetic chosen response"
            }

        # Add to preference pairs
        self.preference_pairs.append(preference_pair)
        self._save_preference_pairs()

    def _generate_synthetic_worse_response(self, query: str, good_response: str) -> str:
        """
        Generate a synthetic worse response to pair with a liked response.

        This is a simple template-based approach. In a production system,
        this would use an LLM to generate realistic worse responses.
        """
        # Simple deterioration of a good response to create a worse one
        worse_templates = [
            "I'm not sure about the specifics, but generally speaking, Australian law addresses this issue. You might want to look up more details on your own.",

            "This is a complex legal question that would require consultation with a lawyer who specializes in this area of Australian law.",

            "The answer to your question depends on many factors that were not specified in your query. Australian law is nuanced and jurisdiction-specific.",

            "There might be some legal principles that apply to your situation, but without more specific information, I can only provide general information about Australian law."
        ]

        import random
        return random.choice(worse_templates)

    def _generate_synthetic_better_response(self, query: str, bad_response: str) -> str:
        """
        Generate a synthetic better response to pair with a disliked response.

        This is a simple template-based approach. In a production system,
        this would use an LLM to generate realistic better responses.
        """
        # Simple templates for better responses
        better_templates = [
            f"To properly answer your question about {query.lower()}, I would need to provide specific references to Australian legislation and case law. The most relevant legal principles would include consideration of jurisdiction-specific regulations and how courts have interpreted them in similar cases.",

            f"Your question about {query.lower()} touches on an important area of Australian law. A comprehensive answer would need to address the relevant statutes, case precedents, and jurisdiction-specific considerations, with proper legal citations and clear distinctions between established legal principles and areas where the law is still evolving.",

            f"Regarding {query.lower()}, Australian law provides specific guidance through legislation such as the relevant Acts and through established case precedents. Any accurate answer would need to cite these sources specifically and distinguish between federal law and state/territory jurisdictions where applicable."
        ]

        import random
        return random.choice(better_templates)

    def create_preference_dataset(self) -> EnhancedLegalPreferenceDataset:
        """
        Create a dataset for training the reward model.

        Returns:
            Dataset object for training
        """
        return EnhancedLegalPreferenceDataset(self.preference_pairs)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback dataset.

        Returns:
            Dictionary with dataset statistics
        """
        # Count positive and negative feedback
        positive_count = sum(1 for item in self.raw_feedback if item["is_positive"])
        negative_count = len(self.raw_feedback) - positive_count

        return {
            "total_raw_feedback": len(self.raw_feedback),
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "total_pairs": len(self.preference_pairs)
        }


class EnhancedLegalRLHF:
    """Enhanced RLHF system for legal responses with like/dislike feedback."""

    def __init__(self, config= None):
        """Initialize RLHF system."""
        # Initialize feedback collector
        self.config = config
        self.feedback_collector = EnhancedLegalFeedbackCollector()

        # Initialize reward model
        self.reward_model = EnhancedLegalRewardModel()

        # Configuration
        self.min_preference_pairs = self.config.get("rlhf", {}).get("feedback", {}).get("min_preference_pairs", 30)
        self.pending_training = False

        # Check if we have enough data to train
        self._check_training_status()

        logger.info("Initialized enhanced RLHF system")

    def _check_training_status(self):
        """Check if we have enough data to train the model."""
        stats = self.feedback_collector.get_dataset_stats()

        # Check if we have enough preference pairs
        if stats["total_pairs"] >= self.min_preference_pairs:
            self.pending_training = True
            logger.info(f"Sufficient preference pairs ({stats['total_pairs']}) for training")
        else:
            self.pending_training = False
            logger.info(f"Need more preference pairs for training ({stats['total_pairs']}/{self.min_preference_pairs})")

    def collect_like_dislike_feedback(self, query: str, response: str, is_positive: bool,
                                      feedback_text: Optional[str] = None):
        """
        Collect like/dislike feedback.

        Args:
            query: The legal query
            response: The response being rated
            is_positive: True for like, False for dislike
            feedback_text: Optional text explaining the rating
        """
        # Collect feedback
        self.feedback_collector.collect_like_dislike_feedback(
            query=query,
            response=response,
            is_positive=is_positive,
            feedback_text=feedback_text
        )

        # Update training status
        self._check_training_status()

    def evaluate_response(self, query: str, response: str) -> float:
        """
        Evaluate a response using the reward model.

        Args:
            query: The legal query
            response: The response to evaluate

        Returns:
            Quality score (higher is better)
        """
        try:
            return self.score_response(query, response)
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return 0.7  # Return a default reasonable score if evaluation fails

    def score_response(self, query: str, response: str) -> float:
        """
        Score a response using the reward model.

        Args:
            query: The legal query
            response: The response to evaluate

        Returns:
            Quality score (higher is better)
        """
        return self.reward_model.score_response(query, response)

    def create_better_response(self, query: str, response: str, context: str, llm_api) -> str:
        """
        Try to create a better response based on reward model.

        Args:
            query: The legal query
            response: The initial response
            context: Retrieved context
            llm_api: LLM API for generating alternatives

        Returns:
            Improved response or original if no improvement
        """
        return self.reward_model.create_better_response(query, response, context, llm_api)

    def run_training_if_needed(self) -> bool:
        """
        Train the reward model if we have enough data.

        Returns:
            True if training was performed, False otherwise
        """
        if not self.pending_training:
            logger.info("Not enough data for training")
            return False

        logger.info("Training reward model from collected feedback")

        # Create dataset
        dataset = self.feedback_collector.create_preference_dataset()

        # Train model
        success = self.reward_model.train(dataset)

        if success:
            self.pending_training = False
            logger.info("Reward model training complete")
            return True
        else:
            logger.error("Reward model training failed")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the RLHF system.

        Returns:
            Dictionary with status information
        """
        dataset_stats = self.feedback_collector.get_dataset_stats()

        return {
            "dataset": dataset_stats,
            "min_preference_pairs": self.min_preference_pairs,
            "pending_training": self.pending_training
        }
