# """
# LegalMind RLHF Implementation
#
# This module implements a Reinforcement Learning from Human Feedback (RLHF)
# system for improving legal response quality.
# """
#
# import json
# import logging
# import os
# import torch
# import numpy as np
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple
# from pathlib import Path
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     Trainer,
#     TrainingArguments
# )
# from torch.utils.data import Dataset
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class FeedbackItem:
#     """Represents a single feedback item for legal responses."""
#
#     def __init__(self, query: str, response: str, is_positive: bool, feedback_text: Optional[str] = None):
#         """Initialize with query, response and rating."""
#         self.query = query
#         self.response = response
#         self.is_positive = is_positive
#         self.feedback_text = feedback_text
#         self.timestamp = datetime.now().isoformat()
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for storage."""
#         return {
#             "query": self.query,
#             "response": self.response,
#             "is_positive": self.is_positive,
#             "feedback_text": self.feedback_text,
#             "timestamp": self.timestamp
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
#         """Create from dictionary."""
#         item = cls(
#             query=data["query"],
#             response=data["response"],
#             is_positive=data["is_positive"],
#             feedback_text=data.get("feedback_text")
#         )
#         item.timestamp = data.get("timestamp", datetime.now().isoformat())
#         return item
#
#
# class PreferenceDataset(Dataset):
#     """Dataset for training a preference model on legal responses."""
#
#     def __init__(self, preference_pairs: List[Dict[str, Any]]):
#         """Initialize with preference pairs."""
#         logger.info(f"Creating dataset with {len(preference_pairs)} preference pairs")
#         self.preference_pairs = preference_pairs
#
#         # Load tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#         self.max_length = 512
#
#     def __len__(self):
#         """Return dataset length."""
#         return len(self.preference_pairs)
#
#     def __getitem__(self, idx):
#         """Get tokenized item."""
#         item = self.preference_pairs[idx]
#
#         # Get query, chosen response and rejected response
#         query = item["query"]
#         chosen = item["chosen"]
#         rejected = item["rejected"]
#
#         # Tokenize chosen response
#         chosen_tokens = self.tokenizer(
#             f"Query: {query}\n\nResponse: {chosen}",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#
#         # Tokenize rejected response
#         rejected_tokens = self.tokenizer(
#             f"Query: {query}\n\nResponse: {rejected}",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#
#         return {
#             "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
#             "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
#             "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
#             "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
#         }
#
#
# # Modified: Custom Trainer with compute_loss method
# class PreferenceTrainer(Trainer):
#     """Custom trainer with preference loss function."""
#
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         # Process chosen responses
#         chosen_outputs = model(
#             input_ids=inputs["chosen_input_ids"],
#             attention_mask=inputs["chosen_attention_mask"]
#         )
#
#         # Process rejected responses
#         rejected_outputs = model(
#             input_ids=inputs["rejected_input_ids"],
#             attention_mask=inputs["rejected_attention_mask"]
#         )
#
#         # Get logits
#         chosen_logits = chosen_outputs.logits
#         rejected_logits = rejected_outputs.logits
#
#         # Calculate preference loss (log sigmoid of the difference)
#         loss = -torch.log(torch.sigmoid(chosen_logits - rejected_logits)).mean()
#
#         return (loss, chosen_outputs) if return_outputs else loss
#
#
# class RewardModel:
#     """Reward model for evaluating legal response quality."""
#
#     def __init__(self, model_path: Optional[str] = None, config=None):
#         """Initialize reward model."""
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.config = config
#
#         # Default model path if not provided
#         if model_path is None:
#             model_path = os.path.join("models", "reward_model")
#         self.model_path = model_path
#
#         if os.path.exists(self.model_path) and os.path.isdir(self.model_path) and os.path.exists(os.path.join(self.model_path, "config.json")):
#             logger.info(f"Loading reward model from {self.model_path}")
#             try:
#                 self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
#                 self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#                 logger.info("Model loaded successfully")
#             except Exception as e:
#                 logger.error(f"Error loading model: {str(e)}")
#                 self._initialize_new_model()
#         else:
#             logger.info("No existing model found. Initializing new model.")
#             self._initialize_new_model()
#
#         self.model.to(self.device)
#         self.max_length = 512
#
#     def _initialize_new_model(self):
#         """Initialize a new reward model."""
#         # Use DistilBERT for a lightweight but effective model
#         base_model = "distilbert-base-uncased"
#         logger.info(f"Initializing new reward model from {base_model}")
#
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             base_model,
#             num_labels=1
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(base_model)
#
#     def save_model(self):
#         """Save the reward model."""
#         os.makedirs(self.model_path, exist_ok=True)
#         self.model.save_pretrained(self.model_path)
#         self.tokenizer.save_pretrained(self.model_path)
#         logger.info(f"Saved reward model to {self.model_path}")
#
#     def score_response(self, query: str, response: str) -> float:
#         """
#         Score a response given a query.
#
#         Args:
#             query: The legal query
#             response: The response to evaluate
#
#         Returns:
#             Reward score (higher is better)
#         """
#         inputs = self.tokenizer(
#             f"Query: {query}\n\nResponse: {response}",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         ).to(self.device)
#
#         # Don't compute gradients
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#
#         # Get the prediction (scalar reward value)
#         reward = outputs.logits.item()
#
#         # Normalize to 0-1 range with sigmoid
#         normalized_reward = 1 / (1 + np.exp(-reward))
#
#         return normalized_reward
#
#     def create_better_response(self, query: str, initial_response: str, context: str, llm_api) -> str:
#         """
#         Try to create a better response based on reward model scoring.
#
#         Args:
#             query: The user query
#             initial_response: The initial LLM response
#             context: The retrieved context
#             llm_api: The LLM API object to generate alternatives
#
#         Returns:
#             The improved response or the original if no improvement
#         """
#         # Score the initial response
#         initial_score = self.score_response(query, initial_response)
#
#         # If already high quality, return as is
#         if initial_score > 0.9:
#             logger.info(f"Initial response already high quality (score: {initial_score:.2f})")
#             return initial_response
#
#         # Try to generate an improved response
#         try:
#             # Create a prompt instructing the model to improve the response
#             improvement_prompt = f"""
#                 You provided this response to a legal query, but it could be improved:
#
#                 QUERY: {query}
#
#                 YOUR RESPONSE:
#                 {initial_response}
#
#                 Please provide an improved response that:
#                 1. Includes more accurate legal citations
#                 2. Is more precise in legal terminology
#                 3. Clearly distinguishes between jurisdictions
#                 4. Avoids definitive statements without support
#
#                 IMPROVED RESPONSE:
#                 """
#
#             # Generate a new response using the improvement prompt and context
#             logger.info("Attempting to generate improved response")
#             improved_response = llm_api.generate(improvement_prompt, context)
#
#             # Score the improved response
#             improved_score = self.score_response(query, improved_response)
#
#             logger.info(f"Initial score: {initial_score:.2f}, Improved score: {improved_score:.2f}")
#
#             # Only use the improved response if it scores better
#             if improved_score > initial_score:
#                 logger.info("Using improved response")
#                 return improved_response
#             else:
#                 logger.info("Keeping original response as improvement not better")
#                 return initial_response
#
#         except Exception as e:
#             logger.error(f"Error generating improved response: {str(e)}")
#             return initial_response
#
#     def train(self, dataset, output_dir=None):
#         """
#         Train the reward model on preference data.
#
#         Args:
#             dataset: Training dataset
#             output_dir: Directory to save model (defaults to self.model_path)
#         """
#         if output_dir is None:
#             output_dir = self.model_path
#
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Define training arguments
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             num_train_epochs=3,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             warmup_steps=100,
#             weight_decay=0.01,
#             logging_dir=os.path.join(output_dir, 'logs'),
#             logging_steps=10,
#             save_steps=50,
#             remove_unused_columns=False,
#         )
#
#         # Create trainer - using our custom trainer class
#         trainer = PreferenceTrainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=dataset,
#         )
#
#         # Train model
#         logger.info("Training reward model...")
#         try:
#             trainer.train()
#
#             # Save model
#             self.save_model()
#             logger.info("Reward model training complete")
#             return True
#         except Exception as e:
#             logger.error(f"Error training model: {str(e)}")
#             return False
#
#
# class FeedbackCollector:
#     """Collects and manages human feedback on legal responses."""
#
#     def __init__(self, feedback_path: Optional[str] = None, config=None):
#         """Initialize feedback collector."""
#         self.config = config
#         self.feedback_path = feedback_path or os.path.join("data", "feedback")
#         os.makedirs(self.feedback_path, exist_ok=True)
#
#         # File for raw feedback data
#         self.raw_feedback_file = os.path.join(self.feedback_path, "legal_feedback.json")
#
#         # File for processed preference pairs
#         self.preference_pairs_file = os.path.join(self.feedback_path, "legal_preference_pairs.json")
#
#         # File for RLHF status
#         self.status_file = os.path.join(self.feedback_path, "rlhf_status.json")
#
#         # Load existing feedback data
#         self.raw_feedback = self._load_raw_feedback()
#         self.preference_pairs = self._load_preference_pairs()
#         self.status = self._load_status()
#
#         logger.info(
#             f"Initialized feedback collector with {len(self.raw_feedback)} items and {len(self.preference_pairs)} preference pairs")
#
#     def _load_raw_feedback(self) -> List[Dict[str, Any]]:
#         """Load raw feedback items."""
#         if os.path.exists(self.raw_feedback_file):
#             try:
#                 with open(self.raw_feedback_file, 'r') as f:
#                     return json.load(f)
#             except json.JSONDecodeError:
#                 logger.error(f"Error decoding {self.raw_feedback_file}")
#                 return []
#         return []
#
#     def _load_preference_pairs(self) -> List[Dict[str, Any]]:
#         """Load preference pairs."""
#         if os.path.exists(self.preference_pairs_file):
#             try:
#                 with open(self.preference_pairs_file, 'r') as f:
#                     return json.load(f)
#             except json.JSONDecodeError:
#                 logger.error(f"Error decoding {self.preference_pairs_file}")
#                 return []
#         return []
#
#     def _load_status(self) -> Dict[str, Any]:
#         """Load RLHF status."""
#         if os.path.exists(self.status_file):
#             try:
#                 with open(self.status_file, 'r') as f:
#                     return json.load(f)
#             except json.JSONDecodeError:
#                 logger.error(f"Error decoding {self.status_file}")
#                 return {"pending_training": False}
#         return {"pending_training": False}
#
#     def _save_raw_feedback(self):
#         """Save raw feedback items."""
#         with open(self.raw_feedback_file, 'w') as f:
#             json.dump(self.raw_feedback, f, indent=2)
#
#     def _save_preference_pairs(self):
#         """Save preference pairs."""
#         with open(self.preference_pairs_file, 'w') as f:
#             json.dump(self.preference_pairs, f, indent=2)
#
#     def _save_status(self, status=None):
#         """Save RLHF status."""
#         if status is None:
#             status = self.status
#         with open(self.status_file, 'w') as f:
#             json.dump(status, f, indent=2)
#
#     def collect_like_dislike_feedback(self, query: str, response: str, is_positive: bool,
#                                       feedback_text: Optional[str] = None):
#         """
#         Collect like/dislike feedback on a response.
#
#         Args:
#             query: The legal query
#             response: The response being rated
#             is_positive: True for like, False for dislike
#             feedback_text: Optional text explaining the rating
#         """
#         # Create feedback item
#         feedback_item = FeedbackItem(
#             query=query,
#             response=response,
#             is_positive=is_positive,
#             feedback_text=feedback_text
#         )
#
#         # Add to raw feedback
#         self.raw_feedback.append(feedback_item.to_dict())
#         self._save_raw_feedback()
#
#         # Create synthetic preference pair
#         self._create_preference_pair_from_feedback(feedback_item)
#
#         logger.info(f"Collected {'positive' if is_positive else 'negative'} feedback for response")
#
#     def _create_preference_pair_from_feedback(self, feedback_item: FeedbackItem):
#         """
#         Create a preference pair from like/dislike feedback.
#
#         For positive feedback, the liked response is chosen.
#         For negative feedback, we create a synthetic better response.
#         """
#         query = feedback_item.query
#         response = feedback_item.response
#
#         if feedback_item.is_positive:
#             # For positive feedback, we need a synthetic "worse" response
#             # This would ideally be generated, but for simplicity we'll use templates
#             rejected_response = self._generate_synthetic_worse_response(query, response)
#
#             preference_pair = {
#                 "query": query,
#                 "chosen": response,
#                 "rejected": rejected_response,
#                 "feedback": "Positive user feedback, synthetic rejected response"
#             }
#
#         else:
#             # For negative feedback, we need a synthetic "better" response
#             chosen_response = self._generate_synthetic_better_response(query, response)
#
#             preference_pair = {
#                 "query": query,
#                 "chosen": chosen_response,
#                 "rejected": response,
#                 "feedback": "Negative user feedback, synthetic chosen response"
#             }
#
#         # Add to preference pairs
#         self.preference_pairs.append(preference_pair)
#         self._save_preference_pairs()
#
#         # Update status
#         if len(self.preference_pairs) >= 10:  # Threshold for pending training
#             self.status["pending_training"] = True
#             self._save_status()
#
#     def _generate_synthetic_worse_response(self, query: str, good_response: str) -> str:
#         """
#         Generate a synthetic worse response to pair with a liked response.
#
#         This is a template-based approach, which could be expanded to use
#         the LLM for more sophisticated alternatives.
#         """
#         # Simple deterioration of a good response to create a worse one
#         worse_templates = [
#             "I'm not sure about the specifics, but generally speaking, Australian law addresses this issue. You might want to look up more details on your own.",
#
#             "This is a complex legal question that would require consultation with a lawyer who specializes in this area of Australian law.",
#
#             "The answer to your question depends on many factors that were not specified in your query. Australian law is nuanced and jurisdiction-specific.",
#
#             "There might be some legal principles that apply to your situation, but without more specific information, I can only provide general information about Australian law."
#         ]
#
#         import random
#         return random.choice(worse_templates)
#
#     def _generate_synthetic_better_response(self, query: str, bad_response: str) -> str:
#         """
#         Generate a synthetic better response to pair with a disliked response.
#
#         This is a template-based approach, which could be expanded to use
#         the LLM for more sophisticated alternatives.
#         """
#         # Simple templates for better responses
#         better_templates = [
#             f"To properly answer your question about {query.lower()}, I would need to provide specific references to Australian legislation and case law. The most relevant legal principles would include consideration of jurisdiction-specific regulations and how courts have interpreted them in similar cases.",
#
#             f"Your question about {query.lower()} touches on an important area of Australian law. A comprehensive answer would need to address the relevant statutes, case precedents, and jurisdiction-specific considerations, with proper legal citations and clear distinctions between established legal principles and areas where the law is still evolving.",
#
#             f"Regarding {query.lower()}, Australian law provides specific guidance through legislation such as the relevant Acts and through established case precedents. Any accurate answer would need to cite these sources specifically and distinguish between federal law and state/territory jurisdictions where applicable."
#         ]
#
#         import random
#         return random.choice(better_templates)
#
#     def generate_synthetic_examples(self, num_examples: int = 50):
#         """
#         Generate synthetic preference data for initial training.
#
#         Args:
#             num_examples: Number of synthetic examples to generate
#         """
#         logger.info(f"Generating {num_examples} synthetic preference examples")
#
#         # Example of synthetic data structure
#         sample_queries = [
#             "What constitutes negligence in Australian law?",
#             "Explain the concept of duty of care.",
#             "How does adverse possession work in Victoria?",
#             "What are the elements of a valid contract?",
#             "What rights do tenants have in New South Wales?"
#         ]
#
#         # Good and bad response templates
#         good_templates = [
#             "Under Australian law, {concept} is defined by {principles}. This is established in the case of {case}, which held that {ruling}. In the jurisdiction of {jurisdiction}, this is particularly relevant because {reason}.",
#             "In {jurisdiction}, the legal framework for {concept} is found in {legislation}. Courts have interpreted this to mean {interpretation}, as seen in {case} where the court found that {ruling}."
#         ]
#
#         bad_templates = [
#             "I believe {concept} means {incorrect_definition}. You should consult a lawyer for more details.",
#             "{concept} is a complex area of law. Different rules might apply in your situation."
#         ]
#
#         import random
#
#         # Sample data to fill templates
#         concepts = ["negligence", "duty of care", "adverse possession", "contract formation", "property rights"]
#         principles = ["a breach of standard of care", "reasonable foreseeability", "continuous possession for 15 years", "offer, acceptance, and consideration", "exclusive possession and control"]
#         cases = ["Smith v Jones [2010] NSWSC 123", "Thompson v Wright [2015] VSC 234", "Wilson v Clark [2012] QSC 345"]
#         jurisdictions = ["New South Wales", "Victoria", "Queensland", "Western Australia", "the Commonwealth"]
#         legislations = ["the Civil Liability Act", "the Property Law Act", "the Contract Law Reform Act", "the Common Law Procedure Act"]
#         rulings = ["failure to take reasonable care constitutes negligence", "a duty of care exists between parties in proximity", "continuous possession must be without permission of the owner"]
#         reasons = ["state legislation has codified the common law principles", "the court system has a distinct approach to these cases", "there are specific statutory provisions that modify the common law"]
#
#         incorrect_definitions = ["breaking the law", "whatever the judge decides", "a way to avoid responsibility"]
#
#         for i in range(num_examples):
#             # Generate random query
#             query = random.choice(sample_queries)
#
#             # Fill templates with random selections
#             replacements = {
#                 "concept": random.choice(concepts),
#                 "principles": random.choice(principles),
#                 "case": random.choice(cases),
#                 "jurisdiction": random.choice(jurisdictions),
#                 "legislation": random.choice(legislations),
#                 "ruling": random.choice(rulings),
#                 "reason": random.choice(reasons),
#                 "incorrect_definition": random.choice(incorrect_definitions),
#                 "interpretation": "the standard of care depends on the specific context"
#             }
#
#             # Format templates
#             good_response = random.choice(good_templates).format(**replacements)
#             bad_response = random.choice(bad_templates).format(**replacements)
#
#             # Create preference pair
#             preference_pair = {
#                 "query": query,
#                 "chosen": good_response,
#                 "rejected": bad_response,
#                 "feedback": "Synthetic training example"
#             }
#
#             self.preference_pairs.append(preference_pair)
#
#         # Save updated preference pairs
#         self._save_preference_pairs()
#
#         # Update status
#         if len(self.preference_pairs) >= 10:  # Threshold for pending training
#             self.status["pending_training"] = True
#             self._save_status()
#
#         logger.info(f"Generated {num_examples} synthetic preference pairs")
#
#     def create_preference_dataset(self) -> PreferenceDataset:
#         """
#         Create a dataset for training the reward model.
#
#         Returns:
#             Dataset object for training
#         """
#         return PreferenceDataset(self.preference_pairs)
#
#     def get_dataset_stats(self) -> Dict[str, Any]:
#         """
#         Get statistics about the feedback dataset.
#
#         Returns:
#             Dictionary with dataset statistics
#         """
#         # Count positive and negative feedback
#         positive_count = sum(1 for item in self.raw_feedback if item["is_positive"])
#         negative_count = len(self.raw_feedback) - positive_count
#
#         return {
#             "total_raw_feedback": len(self.raw_feedback),
#             "positive_feedback": positive_count,
#             "negative_feedback": negative_count,
#             "total_pairs": len(self.preference_pairs)
#         }
#
#     def _load_raw_feedback(self) -> List[Dict[str, Any]]:
#         """Load raw feedback items."""
#         if os.path.exists(self.raw_feedback_file):
#             try:
#                 with open(self.raw_feedback_file, 'r') as f:
#                     data = json.load(f)
#
#                     # Handle data format issues - ensure is_positive exists
#                     for item in data:
#                         if "is_positive" not in item and isinstance(item, dict):
#                             # Try to infer from other fields or default to positive
#                             if "feedback" in item and isinstance(item["feedback"], str) and "positive" in item["feedback"].lower():
#                                 item["is_positive"] = True
#                             else:
#                                 # Default to positive if can't determine
#                                 item["is_positive"] = True
#
#                     return data
#
#             except json.JSONDecodeError:
#                 logger.error(f"Error decoding {self.raw_feedback_file}")
#                 return []
#         return []
#
#
# class LegalRLHF:
#     """RLHF system for legal responses with like/dislike feedback."""
#
#     def __init__(self, config=None):
#         """Initialize RLHF system."""
#         self.config = config
#
#         # Initialize feedback collector
#         self.feedback_collector = FeedbackCollector(config=config)
#
#         # Initialize reward model
#         self.reward_model = RewardModel(config=config)
#
#         # Configuration
#         self.min_preference_pairs = 10  # Default, override from config if available
#         if self.config is not None:
#             self.min_preference_pairs = self.config.get("rlhf", {}).get("feedback", {}).get("min_preference_pairs", 10)
#
#         self.pending_training = self.feedback_collector.status.get("pending_training", False)
#
#         # Check if we have enough data to train
#         self._check_training_status()
#
#         logger.info("Initialized RLHF system")
#
#     def _check_training_status(self):
#         """Check if we have enough data to train the model."""
#         stats = self.feedback_collector.get_dataset_stats()
#
#         # Check if we have enough preference pairs
#         if stats["total_pairs"] >= self.min_preference_pairs:
#             self.pending_training = True
#             self.feedback_collector.status["pending_training"] = True
#             self.feedback_collector._save_status()
#             logger.info(f"Sufficient preference pairs ({stats['total_pairs']}) for training")
#         else:
#             self.pending_training = False
#             self.feedback_collector.status["pending_training"] = False
#             self.feedback_collector._save_status()
#             logger.info(f"Need more preference pairs for training ({stats['total_pairs']}/{self.min_preference_pairs})")
#
#     def collect_like_dislike_feedback(self, query: str, response: str, is_positive: bool,
#                                       feedback_text: Optional[str] = None):
#         """
#         Collect like/dislike feedback.
#
#         Args:
#             query: The legal query
#             response: The response being rated
#             is_positive: True for like, False for dislike
#             feedback_text: Optional text explaining the rating
#         """
#         # Collect feedback
#         self.feedback_collector.collect_like_dislike_feedback(
#             query=query,
#             response=response,
#             is_positive=is_positive,
#             feedback_text=feedback_text
#         )
#
#         # Update training status
#         self._check_training_status()
#
#     def evaluate_response(self, query: str, response: str) -> float:
#         """
#         Evaluate a response using the reward model.
#
#         Args:
#             query: The legal query
#             response: The response to evaluate
#
#         Returns:
#             Quality score (higher is better)
#         """
#         try:
#             return self.score_response(query, response)
#         except Exception as e:
#             logger.error(f"Error evaluating response: {str(e)}")
#             return 0.7  # Return a default reasonable score if evaluation fails
#
#     def score_response(self, query: str, response: str) -> float:
#         """
#         Score a response using the reward model.
#
#         Args:
#             query: The legal query
#             response: The response to evaluate
#
#         Returns:
#             Quality score (higher is better)
#         """
#         return self.reward_model.score_response(query, response)
#
#     def create_better_response(self, query: str, response: str, context: str, llm_api) -> str:
#         """
#         Try to create a better response based on reward model.
#
#         Args:
#             query: The legal query
#             response: The initial response
#             context: Retrieved context
#             llm_api: LLM API for generating alternatives
#
#         Returns:
#             Improved response or original if no improvement
#         """
#         return self.reward_model.create_better_response(query, response, context, llm_api)
#
#     def run_training_if_needed(self) -> bool:
#         """
#         Train the reward model if we have enough data.
#
#         Returns:
#             True if training was performed, False otherwise
#         """
#         if not self.pending_training:
#             logger.info("Not enough data for training")
#             return False
#
#         logger.info("Training reward model from collected feedback")
#
#         # Create dataset
#         dataset = self.feedback_collector.create_preference_dataset()
#
#         # Train model
#         success = self.reward_model.train(dataset)
#
#         if success:
#             self.pending_training = False
#             self.feedback_collector.status["pending_training"] = False
#             self.feedback_collector._save_status()
#             logger.info("Reward model training complete")
#             return True
#         else:
#             logger.error("Reward model training failed")
#             return False
#
#     def initialize_with_synthetic_preferences(self, num_examples: int = 50) -> bool:
#         """
#         Initialize with synthetic preference data for bootstrapping.
#         """
#         try:
#             # Check if we already have preference pairs
#             if len(self.feedback_collector.preference_pairs) > 0:
#                 logger.info(f"Skipping synthetic examples generation: {len(self.feedback_collector.preference_pairs)} preference pairs already exist")
#                 return True
#
#             # Generate synthetic examples
#             self.feedback_collector.generate_synthetic_examples(num_examples)
#
#             # Update training status
#             self._check_training_status()
#
#             # Train model if we have enough data
#             if self.pending_training:
#                 return self.run_training_if_needed()
#
#             return True
#         except Exception as e:
#             logger.error(f"Error initializing with synthetic preferences: {str(e)}")
#             return False
#
#     def get_status(self) -> Dict[str, Any]:
#         """
#         Get status information about the RLHF system.
#
#         Returns:
#             Dictionary with status information
#         """
#         dataset_stats = self.feedback_collector.get_dataset_stats()
#
#         return {
#             "dataset": dataset_stats,
#             "min_preference_pairs": self.min_preference_pairs,
#             "pending_training": self.pending_training
#         }
"""
LegalMind RLHF Implementation

This module implements a Reinforcement Learning from Human Feedback (RLHF)
system for improving legal response quality with specialized features for
legal reasoning and Australian law contexts.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

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


class FeedbackItem:
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
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create from dictionary."""
        item = cls(
            query=data["query"],
            response=data["response"],
            is_positive=data["is_positive"],
            feedback_text=data.get("feedback_text")
        )
        item.timestamp = data.get("timestamp", datetime.now().isoformat())
        return item


class PreferenceDataset(Dataset):
    """Dataset for training a preference model on legal responses."""

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


class PreferenceTrainer(Trainer):
    """Custom trainer with preference loss function for legal response quality."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss for preference learning."""
        # Process chosen responses
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        )

        # Process rejected responses
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        )

        # Get logits
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits

        # Calculate preference loss (log sigmoid of the difference)
        loss = -torch.log(torch.sigmoid(chosen_logits - rejected_logits)).mean()

        return (loss, chosen_outputs) if return_outputs else loss


class RewardModel:
    """Reward model for evaluating legal response quality with specialized legal criteria."""

    def __init__(self, model_path: Optional[str] = None, config=None):
        """Initialize reward model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        # Default model path if not provided
        if model_path is None:
            model_path = os.path.join("models", "legal_reward_model")
        self.model_path = model_path

        if os.path.exists(self.model_path) and os.path.isdir(self.model_path) and os.path.exists(os.path.join(self.model_path, "config.json")):
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
                5. Cites Australian case law and legislation where appropriate
                
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

    def calculate_legal_response_confidence(self, response: str, context: str, citation_checks: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence in a legal response with specialized legal criteria.

        Args:
            response: Generated response
            context: Retrieved context
            citation_checks: Results of citation verification

        Returns:
            Confidence score (0.0-1.0)
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # Initialize semantic similarity model if not available
        if not hasattr(self, 'semantic_model'):
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Check jurisdiction consistency
        jurisdiction_score = self._check_jurisdiction_consistency(response, context)

        # Calculate citation confidence
        citation_scores = [check["confidence"] for check in citation_checks]
        citation_score = sum(citation_scores) / len(citation_scores) if citation_scores else 1.0

        # Calculate semantic similarity between response and context
        if not context.strip():
            # If context is empty, we can't calculate semantic similarity
            semantic_score = 0.5  # Use a neutral score
            logger.warning("Empty context provided for semantic similarity calculation")
        else:
            try:
                response_embedding = self.semantic_model.encode([response])[0]

                # Split context into chunks to handle large contexts
                context_chunks = [context[i:i+1000] for i in range(0, len(context), 1000)]

                # Only proceed if we have context chunks
                if not context_chunks:
                    semantic_score = 0.5
                else:
                    chunk_embeddings = self.semantic_model.encode(context_chunks)

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

    def _check_jurisdiction_consistency(self, response: str, context: str) -> float:
        """
        Check if response is consistent with jurisdictions in context.

        Args:
            response: Generated response
            context: Retrieved context

        Returns:
            Jurisdiction consistency score (0.0-1.0)
        """
        # Jurisdiction keywords mapping
        jurisdiction_keywords = {
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

        # Identify jurisdictions in response and context
        response_jurs = self._identify_jurisdictions(response, jurisdiction_keywords)
        context_jurs = self._identify_jurisdictions(context, jurisdiction_keywords)

        # Check for jurisdictions in response not supported by context
        mismatched_jurs = []
        for jur, score in response_jurs.items():
            if jur not in context_jurs or context_jurs[jur] < score / 2:
                mismatched_jurs.append(jur)

        # Calculate confidence score
        if mismatched_jurs:
            # Reduce confidence proportionally to mismatch
            mismatch_scores = [response_jurs[jur] for jur in mismatched_jurs]
            jurisdiction_score = max(0.0, 1.0 - sum(mismatch_scores) / len(mismatch_scores))
        else:
            jurisdiction_score = 1.0

        return jurisdiction_score

    def _identify_jurisdictions(self, text: str, jurisdiction_keywords: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Identify jurisdictions mentioned in text.

        Args:
            text: Text to analyze
            jurisdiction_keywords: Mapping of jurisdiction codes to keywords

        Returns:
            Dictionary of jurisdictions and their confidence scores
        """
        text_lower = text.lower()
        scores = {}

        # Check for mentions of each jurisdiction
        for jur, keywords in jurisdiction_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                scores[jur] = min(1.0, count / 3)  # Cap at 1.0

        return scores

    def train(self, dataset, output_dir=None):
        """
        Train the reward model on preference data.

        Args:
            dataset: Training dataset
            output_dir: Directory to save model (defaults to self.model_path)
        """
        if output_dir is None:
            output_dir = self.model_path

        os.makedirs(output_dir, exist_ok=True)

        # Define training arguments
        config_params = {}
        if self.config:
            config_params = self.config.get("rlhf", {}).get("reward_model", {})

        # Ensure parameters are of the correct type
        epochs = float(config_params.get("training_epochs", 3))
        learning_rate = float(config_params.get("learning_rate", 2e-5))
        batch_size = int(config_params.get("batch_size", 8))

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            save_steps=50,
            remove_unused_columns=False,
            learning_rate=learning_rate
        )

        # Create trainer - using our custom trainer class
        trainer = PreferenceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        # Train model
        logger.info("Training reward model...")
        try:
            trainer.train()

            # Save model
            self.save_model()
            logger.info("Reward model training complete")
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False


class FeedbackCollector:
    """Collects and manages human feedback on legal responses."""

    def __init__(self, feedback_path: Optional[str] = None, config=None):
        """Initialize feedback collector."""
        self.config = config

        # Set feedback path from config if provided
        if config:
            feedback_dir = config.get("rlhf", {}).get("feedback", {}).get("feedback_path", "data/feedback")
            if feedback_path is None:
                feedback_path = feedback_dir

        self.feedback_path = feedback_path or os.path.join("data", "feedback")
        os.makedirs(self.feedback_path, exist_ok=True)

        # File for raw feedback data
        self.raw_feedback_file = os.path.join(self.feedback_path, "legal_feedback.json")

        # File for processed preference pairs
        self.preference_pairs_file = os.path.join(self.feedback_path, "legal_preference_pairs.json")

        # File for RLHF status
        self.status_file = os.path.join(self.feedback_path, "rlhf_status.json")

        # Load existing feedback data
        self.raw_feedback = self._load_raw_feedback()
        self.preference_pairs = self._load_preference_pairs()
        self.status = self._load_status()

        logger.info(
            f"Initialized feedback collector with {len(self.raw_feedback)} items and {len(self.preference_pairs)} preference pairs")

    def _load_raw_feedback(self) -> List[Dict[str, Any]]:
        """Load raw feedback items with error handling."""
        if os.path.exists(self.raw_feedback_file):
            try:
                with open(self.raw_feedback_file, 'r') as f:
                    data = json.load(f)

                    # Handle data format issues - ensure is_positive exists
                    for item in data:
                        if "is_positive" not in item and isinstance(item, dict):
                            # Try to infer from other fields or default to positive
                            if "feedback" in item and isinstance(item["feedback"], str) and "positive" in item["feedback"].lower():
                                item["is_positive"] = True
                            else:
                                # Default to positive if can't determine
                                item["is_positive"] = True

                    return data

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

    def _load_status(self) -> Dict[str, Any]:
        """Load RLHF status."""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding {self.status_file}")
                return {"pending_training": False}
        return {"pending_training": False}

    def _save_raw_feedback(self):
        """Save raw feedback items."""
        with open(self.raw_feedback_file, 'w') as f:
            json.dump(self.raw_feedback, f, indent=2)

    def _save_preference_pairs(self):
        """Save preference pairs."""
        with open(self.preference_pairs_file, 'w') as f:
            json.dump(self.preference_pairs, f, indent=2)

    def _save_status(self, status=None):
        """Save RLHF status."""
        if status is None:
            status = self.status
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

    # Public method for external access
    def save_status(self, status=None):
        """Public method to save RLHF status."""
        self._save_status(status)

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
        feedback_item = FeedbackItem(
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

    def _create_preference_pair_from_feedback(self, feedback_item: FeedbackItem):
        """
        Create a preference pair from like/dislike feedback.

        For positive feedback, the liked response is chosen.
        For negative feedback, we create a synthetic better response.
        """
        query = feedback_item.query
        response = feedback_item.response

        if feedback_item.is_positive:
            # For positive feedback, we need a synthetic "worse" response
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

        # Check if we have enough preference pairs for training
        min_pairs = 10
        if self.config:
            min_pairs = int(self.config.get("rlhf", {}).get("feedback", {}).get("min_preference_pairs", 10))

        if len(self.preference_pairs) >= min_pairs:
            self.status["pending_training"] = True
            self._save_status()

    def _generate_synthetic_worse_response(self, query: str, good_response: str) -> str:
        """
        Generate a synthetic worse response to pair with a liked response.

        This is a template-based approach, which could be expanded to use
        the LLM for more sophisticated alternatives.
        """
        # Simple deterioration of a good response to create a worse one
        worse_templates = [
            "I'm not sure about the specifics, but generally speaking, Australian law addresses this issue. You might want to look up more details on your own.",

            "This is a complex legal question that would require consultation with a lawyer who specializes in this area of Australian law.",

            "The answer to your question depends on many factors that were not specified in your query. Australian law is nuanced and jurisdiction-specific.",

            "There might be some legal principles that apply to your situation, but without more specific information, I can only provide general information about Australian law.",

            "I believe the law says something about this topic, but I don't recall the specific details. It would be best to consult a legal professional.",

            "The legal situation regarding your question has changed in recent years, making it difficult to give a definitive answer without more research."
        ]

        import random
        return random.choice(worse_templates)

    def _generate_synthetic_better_response(self, query: str, bad_response: str) -> str:
        """
        Generate a synthetic better response to pair with a disliked response.

        This is a template-based approach, which could be expanded to use
        the LLM for more sophisticated alternatives.
        """
        # Simple templates for better responses
        better_templates = [
            f"To properly answer your question about {query.lower()}, I would need to provide specific references to Australian legislation and case law. The most relevant legal principles would include consideration of jurisdiction-specific regulations and how courts have interpreted them in similar cases.",

            f"Your question about {query.lower()} touches on an important area of Australian law. A comprehensive answer would need to address the relevant statutes, case precedents, and jurisdiction-specific considerations, with proper legal citations and clear distinctions between established legal principles and areas where the law is still evolving.",

            f"Regarding {query.lower()}, Australian law provides specific guidance through legislation such as the relevant Acts and through established case precedents. Any accurate answer would need to cite these sources specifically and distinguish between federal law and state/territory jurisdictions where applicable.",

            f"In Australian law, {query.lower()} is governed by both statutory provisions and common law principles. The key legislation includes the relevant Acts, while leading cases such as Smith v Jones [2010] HCA 1 have established important precedents that courts follow when addressing similar issues.",

            f"When considering {query.lower()}, it's important to understand that Australian legal principles in this area derive from multiple sources, including legislation, common law precedents, and regulatory frameworks. The jurisdiction matters significantly, as laws can vary between states and territories."
        ]

        import random
        return random.choice(better_templates)

    def generate_synthetic_examples(self, num_examples: int = 50):
        """
        Generate synthetic preference data for initial training.

        Args:
            num_examples: Number of synthetic examples to generate
        """
        logger.info(f"Generating {num_examples} synthetic preference examples")

        # Example of synthetic data structure
        sample_queries = [
            "What constitutes negligence in Australian law?",
            "Explain the concept of duty of care.",
            "How does adverse possession work in Victoria?",
            "What are the elements of a valid contract?",
            "What rights do tenants have in New South Wales?",
            "What legal remedies are available for breach of contract?",
            "How is property divided in divorce cases in Australia?",
            "What is the process for challenging a will in Queensland?",
            "What are my rights as a consumer under Australian law?",
            "How does the Fair Work Act protect employees in Australia?"
        ]

        # Good and bad response templates
        good_templates = [
            "Under Australian law, {concept} is defined by {principles}. This is established in the case of {case}, which held that {ruling}. In the jurisdiction of {jurisdiction}, this is particularly relevant because {reason}.",

            "In {jurisdiction}, the legal framework for {concept} is found in {legislation}. Courts have interpreted this to mean {interpretation}, as seen in {case} where the court found that {ruling}.",

            "Australian law approaches {concept} through a combination of statutory provisions and case law. The {legislation} establishes the basic framework, while {case} provides important guidance on {interpretation}. This is particularly relevant in {jurisdiction} due to {reason}.",

            "The concept of {concept} in Australian law requires consideration of several elements: {principles}. The leading authority is {case}, where the court emphasized that {ruling}. This applies across Australia but has specific implications in {jurisdiction} due to {reason}.",

            "When analyzing {concept} under Australian law, legal practitioners refer to {legislation} as the primary source of authority. This has been interpreted by the courts, particularly in {case}, to mean that {ruling}. The practical application in {jurisdiction} requires consideration of {reason}."
        ]

        bad_templates = [
            "I believe {concept} means {incorrect_definition}. You should consult a lawyer for more details.",

            "{concept} is a complex area of law. Different rules might apply in your situation.",

            "The law regarding {concept} varies from case to case, so it's hard to give a definitive answer. It depends on many factors.",

            "I'm not familiar with the specifics of {concept} in Australian law, but generally speaking, it's related to {incorrect_definition}.",

            "There are laws about {concept} but they change frequently. You should check the latest regulations.",

            "I think {concept} is governed by common law principles, but I'm not sure about the specific details or relevant cases."
        ]

        import random

        # Sample data to fill templates
        concepts = ["negligence", "duty of care", "adverse possession", "contract formation", "property rights",
                    "breach of contract", "defamation", "consumer protection", "unfair dismissal", "tenancy rights"]

        principles = ["a breach of standard of care", "reasonable foreseeability", "continuous possession for 15 years",
                      "offer, acceptance, and consideration", "exclusive possession and control",
                      "clear intention to create legal relations", "substantial performance",
                      "proof of damage to reputation", "misleading or deceptive conduct", "reasonable notice periods"]

        cases = ["Smith v Jones [2010] NSWSC 123", "Thompson v Wright [2015] VSC 234", "Wilson v Clark [2012] QSC 345",
                 "Rogers v Whitaker [1992] HCA 58", "Donoghue v Stevenson [1932] AC 562", "Mabo v Queensland (No 2) [1992] HCA 23",
                 "Waltons Stores v Maher [1988] HCA 7", "Woolcock Street Investments v CDG [2004] HCA 16"]

        jurisdictions = ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia",
                         "Tasmania", "Northern Territory", "Australian Capital Territory", "the Commonwealth"]

        legislations = ["the Civil Liability Act", "the Property Law Act", "the Contract Law Reform Act",
                        "the Common Law Procedure Act", "the Australian Consumer Law", "the Fair Work Act",
                        "the Family Law Act", "the Residential Tenancies Act"]

        rulings = ["failure to take reasonable care constitutes negligence",
                   "a duty of care exists between parties in proximity",
                   "continuous possession must be without permission of the owner",
                   "consideration must have economic value, even if minimal",
                   "misrepresentation can void a contract if material",
                   "implied terms must be necessary for business efficacy",
                   "contributory negligence may reduce damages proportionally"]

        reasons = ["state legislation has codified the common law principles",
                   "the court system has a distinct approach to these cases",
                   "there are specific statutory provisions that modify the common law",
                   "recent amendments have changed the application of these principles",
                   "local precedents have established a unique interpretation",
                   "regulatory bodies have additional enforcement powers"]

        incorrect_definitions = ["breaking the law", "whatever the judge decides", "a way to avoid responsibility",
                                 "similar to what happens in American TV shows", "mostly about paperwork",
                                 "something only rich people need to worry about", "whatever seems fair"]

        interpretations = ["the standard of care depends on the specific context",
                           "prior notice is required in most circumstances",
                           "implied terms can be just as binding as express terms",
                           "reasonable time frames depend on industry standards",
                           "special consideration is given to vulnerable parties",
                           "documentation requirements are strictly enforced"]

        for i in range(num_examples):
            # Generate random query
            query = random.choice(sample_queries)

            # Fill templates with random selections
            replacements = {
                "concept": random.choice(concepts),
                "principles": random.choice(principles),
                "case": random.choice(cases),
                "jurisdiction": random.choice(jurisdictions),
                "legislation": random.choice(legislations),
                "ruling": random.choice(rulings),
                "reason": random.choice(reasons),
                "incorrect_definition": random.choice(incorrect_definitions),
                "interpretation": random.choice(interpretations)
            }

            # Format templates
            good_response = random.choice(good_templates).format(**replacements)
            bad_response = random.choice(bad_templates).format(**replacements)

            # Create preference pair
            preference_pair = {
                "query": query,
                "chosen": good_response,
                "rejected": bad_response,
                "feedback": "Synthetic training example"
            }

            self.preference_pairs.append(preference_pair)

        # Save updated preference pairs
        self._save_preference_pairs()

        # Update status
        min_pairs = 10
        if self.config:
            min_pairs = self.config.get("rlhf", {}).get("feedback", {}).get("min_preference_pairs", 10)

        if len(self.preference_pairs) >= min_pairs:
            self.status["pending_training"] = True
            self._save_status()

        logger.info(f"Generated {num_examples} synthetic preference pairs")

    def create_preference_dataset(self) -> PreferenceDataset:
        """
        Create a dataset for training the reward model.

        Returns:
            Dataset object for training
        """
        return PreferenceDataset(self.preference_pairs)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback dataset.

        Returns:
            Dictionary with dataset statistics
        """
        # Count positive and negative feedback
        positive_count = sum(1 for item in self.raw_feedback if item["is_positive"])
        negative_count = len(self.raw_feedback) - positive_count

        # Get min_preference_pairs from config
        min_pairs = 10
        if self.config:
            min_pairs = int(self.config.get("rlhf", {}).get("feedback", {}).get("min_preference_pairs", 10))

        return {
            "total_raw_feedback": len(self.raw_feedback),
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "total_pairs": len(self.preference_pairs),
            "min_pairs_needed": min_pairs,
            "training_ready": len(self.preference_pairs) >= min_pairs
        }


class LegalRLHF:
    """RLHF system for legal responses with enhanced features for Australian legal contexts."""

    def __init__(self, config=None):
        """Initialize RLHF system."""
        self.config = config

        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector(config=config)

        # Initialize reward model
        self.reward_model = RewardModel(config=config)

        # Configuration
        self.min_preference_pairs = 10
        if self.config is not None:
            self.min_preference_pairs = int(self.config.get("rlhf", {}).get("feedback", {}).get("min_preference_pairs", 10))

        self.pending_training = self.feedback_collector.status.get("pending_training", False)

        # Check if we have enough data to train
        self._check_training_status()

        logger.info("Initialized legal RLHF system")

    def _check_training_status(self):
        """Check if we have enough data to train the model."""
        stats = self.feedback_collector.get_dataset_stats()

        # Check if we have enough preference pairs
        if stats["total_pairs"] >= self.min_preference_pairs:
            self.pending_training = True
            self.feedback_collector.status["pending_training"] = True
            self.feedback_collector._save_status()
            logger.info(f"Sufficient preference pairs ({stats['total_pairs']}) for training")
        else:
            self.pending_training = False
            self.feedback_collector.status["pending_training"] = False
            self.feedback_collector.save_status()
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
            self.feedback_collector.status["pending_training"] = False
            self.feedback_collector.save_status()
            logger.info("Reward model training complete")
            return True
        else:
            logger.error("Reward model training failed")
            return False

    def initialize_with_synthetic_preferences(self, num_examples: int = 50) -> bool:
        """
        Initialize with synthetic preference data for bootstrapping the RLHF system.

        Args:
            num_examples: Number of synthetic examples to generate

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Check if we already have preference pairs
            if len(self.feedback_collector.preference_pairs) > 0:
                logger.info(f"Skipping synthetic examples generation: {len(self.feedback_collector.preference_pairs)} preference pairs already exist")
                return True

            # Generate synthetic examples
            self.feedback_collector.generate_synthetic_examples(num_examples)

            # Update training status
            self._check_training_status()

            # Train model if we have enough data
            if self.pending_training:
                return self.run_training_if_needed()

            return True
        except Exception as e:
            logger.error(f"Error initializing with synthetic preferences: {str(e)}")
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
            "pending_training": self.pending_training,
            "model_initialized": os.path.exists(self.reward_model.model_path)
        }