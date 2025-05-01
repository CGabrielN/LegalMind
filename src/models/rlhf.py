"""
LegalMind RLHF Implementation

This module implements a lightweight Reinforcement Learning from Human Feedback (RLHF)
approach for improving the legal response quality.
"""

import os
import yaml
import json
import torch
import logging
import numpy as np
import pandas as pd  # Added missing pandas import
from datetime import datetime  # Added datetime import for timestamp
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class LegalPreferenceDataset(Dataset):
    """Dataset for training a preference model on legal responses."""

    def __init__(self, data_path: str):
        """Initialize with data path."""
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = 512

        # Load dataset
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} preference pairs")

    def __len__(self):
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get tokenized item."""
        item = self.data[idx]

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

class LegalRewardModel:
    """Reward model for evaluating legal response quality."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize reward model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading reward model from {model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            logger.info("Initializing new reward model from DistilBERT")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=1
            )
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.model.to(self.device)
        self.max_length = 512

    def save_model(self, output_path: str):
        """Save the reward model."""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Saved reward model to {output_path}")

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

        return reward

    def train(self, train_dataset, eval_dataset=None, output_dir="./reward_model"):
        """
        Train the reward model on preference data.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Directory to save model
        """
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
        )

        # Define custom loss function
        def compute_loss(model, inputs):
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

            # Calculate preference loss
            loss = -torch.log(torch.sigmoid(chosen_logits - rejected_logits)).mean()

            return loss

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_loss=compute_loss
        )

        # Train model
        logger.info("Training reward model...")
        trainer.train()

        # Save model
        self.save_model(output_dir)
        logger.info("Reward model training complete")

class LegalFeedbackCollector:
    """Collects and manages human feedback on legal responses."""

    def __init__(self, feedback_path: str = "data/feedback"):
        """Initialize feedback collector."""
        self.feedback_path = feedback_path
        os.makedirs(feedback_path, exist_ok=True)

        # Try to load existing feedback
        self.feedback_file = os.path.join(feedback_path, "legal_feedback.json")
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                self.feedback_data = json.load(f)
        else:
            self.feedback_data = []

        logger.info(f"Initialized feedback collector with {len(self.feedback_data)} entries")

    def add_feedback(self, query: str, chosen_response: str, rejected_response: str, feedback_text: Optional[str] = None):
        """
        Add a new feedback entry.

        Args:
            query: The legal query
            chosen_response: The preferred response
            rejected_response: The less preferred response
            feedback_text: Optional feedback explanation
        """
        feedback_entry = {
            "query": query,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "timestamp": str(datetime.now()),
            "feedback": feedback_text
        }

        self.feedback_data.append(feedback_entry)

        # Save updated feedback
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

        logger.info(f"Added new feedback entry (total: {len(self.feedback_data)})")

    def generate_training_data(self, output_path: Optional[str] = None):
        """
        Convert feedback to training data for reward model.

        Args:
            output_path: Optional path to save training data

        Returns:
            Training dataset
        """
        if not output_path:
            output_path = os.path.join(self.feedback_path, "preference_data.json")

        # Save preference data
        with open(output_path, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

        logger.info(f"Generated training data with {len(self.feedback_data)} preference pairs")
        return self.feedback_data

class LegalRLHF:
    """Main RLHF controller for legal responses."""

    def __init__(self):
        """Initialize RLHF components."""
        # Initialize feedback collector
        self.feedback_collector = LegalFeedbackCollector()

        # Initialize reward model
        reward_model_path = os.path.join("models", "reward_model")
        if os.path.exists(reward_model_path):
            self.reward_model = LegalRewardModel(reward_model_path)
        else:
            self.reward_model = LegalRewardModel()

        logger.info("Initialized RLHF components")

    def evaluate_response(self, query: str, response: str) -> float:
        """
        Evaluate a response using the reward model.

        Args:
            query: The legal query
            response: The response to evaluate

        Returns:
            Quality score (higher is better)
        """
        return self.reward_model.score_response(query, response)

    def collect_feedback(self, query: str, responses: List[str], chosen_idx: int, feedback: Optional[str] = None):
        """
        Collect human feedback on responses.

        Args:
            query: The legal query
            responses: List of generated responses
            chosen_idx: Index of the chosen response
            feedback: Optional feedback text
        """
        # Get chosen response
        chosen = responses[chosen_idx]

        # Get rejected responses (all non-chosen)
        for i, rejected in enumerate(responses):
            if i != chosen_idx:
                self.feedback_collector.add_feedback(query, chosen, rejected, feedback)

        logger.info(f"Collected feedback for query: {query[:50]}...")

    def train_from_feedback(self):
        """Train the reward model from collected feedback."""
        # Generate training data
        preference_data_path = os.path.join("data", "feedback", "preference_data.json")
        self.feedback_collector.generate_training_data(preference_data_path)

        # Create dataset
        train_dataset = LegalPreferenceDataset(preference_data_path)

        # Train model
        os.makedirs("models/reward_model", exist_ok=True)
        self.reward_model.train(train_dataset, output_dir="models/reward_model")

        logger.info("Completed reward model training")

    def initialize_with_synthetic_preferences(self, num_examples: int = 50):
        """
        Initialize with synthetic preference data for bootstrapping.

        Args:
            num_examples: Number of synthetic examples to generate
        """
        # This is just a placeholder for demonstration
        # In a real implementation, you'd generate meaningful legal examples
        logger.info(f"Generating {num_examples} synthetic preference examples")

        # Example of synthetic data structure
        sample_queries = [
            "What constitutes negligence in Australian law?",
            "Explain the concept of duty of care.",
            "How does adverse possession work in Victoria?",
            "What are the elements of a valid contract?",
            "What rights do tenants have in New South Wales?"
        ]

        import random

        for i in range(num_examples):
            query = random.choice(sample_queries)

            # Generate a good response (would be better in real implementation)
            good_response = "This is a comprehensive response with proper legal citations."

            # Generate a worse response
            bad_response = "This is a response without proper legal citations."

            # Add to feedback
            self.feedback_collector.add_feedback(
                query=query,
                chosen_response=good_response,
                rejected_response=bad_response,
                feedback_text="Synthetic example preferring responses with citations"
            )

        logger.info("Generated synthetic preference data")

def create_preference_dataset():
    """Create a basic preference dataset for the legal domain."""
    import pandas as pd

    # Directory for preference data
    os.makedirs("data/feedback", exist_ok=True)

    # Example preference pairs
    preferences = []

    # Example 1
    preferences.append({
        "query": "What constitutes negligence in Australian law?",
        "chosen": "In Australian law, negligence has three key elements: a duty of care, breach of that duty, and resulting damage. For a duty of care to exist, harm must be reasonably foreseeable. In Donoghue v Stevenson [1932] AC 562, the court established the 'neighbor principle' which has been applied in Australian cases such as Jaensch v Coffey (1984) 155 CLR 549. The standard of care expected is that of a reasonable person in the defendant's position.",
        "rejected": "Negligence in Australia means someone was careless. If someone is careless and hurts you, you can sue them. You need to prove they had a duty to be careful, they weren't careful, and you got hurt because of it. Courts look at whether a reasonable person would have acted differently.",
        "feedback": "The chosen response includes specific legal citations and explains legal principles clearly"
    })

    # Example 2
    preferences.append({
        "query": "Explain the legal concept of adverse possession in Victoria.",
        "chosen": "Adverse possession in Victoria allows a person to claim ownership of land they have possessed continuously for at least 15 years without the owner's permission, as established in the Limitations of Actions Act 1958 (Vic). The possession must be actual, open, notorious, exclusive, continuous, and hostile to the true owner's title. In Whittlesea City Council v Abbatangelo [2009] VSCA 188, the Victorian Court of Appeal clarified that the possessor must demonstrate an intention to possess the land to the exclusion of others, including the true owner.",
        "rejected": "If you use someone else's land in Victoria for long enough, you can take it. You need to use it openly for many years. The government might let you claim it if the real owner doesn't complain. It's called adverse possession and happens a lot with boundary disputes between neighbors.",
        "feedback": "The chosen response includes relevant statute, time period, and case law"
    })

    # Example 3
    preferences.append({
        "query": "What are the requirements for a valid will in Queensland?",
        "chosen": "In Queensland, the requirements for a valid will are governed by the Succession Act 1981 (Qld). For a will to be valid, it must be: (1) in writing, (2) signed by the testator or by someone else in the testator's presence and at their direction, (3) the signature must be made with the intention of executing the will, (4) the signature must be witnessed by two or more witnesses present at the same time, and (5) at least two of the witnesses must attest and sign the will in the presence of the testator. As established in Banks v Goodfellow (1870) LR 5 QB 549, the testator must also have testamentary capacity, meaning they understand the nature and effect of making a will, the extent of their property, and the claims to which they should give effect.",
        "rejected": "To make a valid will in Queensland, you need to write it down and sign it. You also need witnesses. The will should say who gets your stuff when you die. You need to be of sound mind when you make it. If you don't make a valid will, the government decides who gets your property.",
        "feedback": "The chosen response cites specific legislation, lists requirements clearly, and mentions relevant case law"
    })

    # Save to JSON file
    with open("data/feedback/preference_data.json", 'w') as f:
        json.dump(preferences, f, indent=2)

    logger.info(f"Created preference dataset with {len(preferences)} examples")
    return "data/feedback/preference_data.json"