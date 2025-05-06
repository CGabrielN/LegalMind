"""
LegalMind Data Ingestion

This module handles data loading and processing from Hugging Face datasets
for Australian legal documents.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional

import yaml
from datasets import load_dataset, Dataset
from tqdm import tqdm

from .chunking import process_legal_document, Chunk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class LegalDataIngestion:
    """Handles ingestion of legal datasets from Hugging Face."""

    def __init__(self, dataset_name: Optional[str] = None):
        """Initialize with dataset name from config or parameter."""
        self.dataset_name = dataset_name or config["dataset"]["name"]
        self.local_path = config["dataset"]["local_path"]

        # Create directory if it doesn't exist
        os.makedirs(self.local_path, exist_ok=True)

    def load_dataset(self) -> Dataset:
        """Load the dataset from Hugging Face."""
        logger.info(f"Loading dataset: {self.dataset_name}")

        try:
            dataset = load_dataset(self.dataset_name)
            logger.info(f"Successfully loaded dataset with {len(dataset['train'])} training examples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset {self.dataset_name}: {str(e)}")

            # Try alternative dataset if specified
            if config["dataset"].get("alternative"):
                alternative = config["dataset"]["alternative"]
                logger.info(f"Trying alternative dataset: {alternative}")
                try:
                    dataset = load_dataset(alternative)
                    logger.info(
                        f"Successfully loaded alternative dataset with {len(dataset['train'])} training examples")
                    self.dataset_name = alternative
                    return dataset
                except Exception as e2:
                    logger.error(f"Error loading alternative dataset: {str(e2)}")

            raise Exception(f"Failed to load any dataset: {str(e)}")

    def extract_legal_documents(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Extract legal documents from the dataset with basic metadata."""
        logger.info("Extracting legal documents from dataset")

        documents = []

        # Different datasets have different structures, handle both cases
        if self.dataset_name == "isaacus/open-australian-legal-qa":
            # This dataset likely has Q&A pairs with legal context
            for item in tqdm(dataset["train"], desc="Processing documents"):
                # Check for context field which should contain the legal text
                if "context" in item and item["context"]:
                    doc = {
                        "text": item["context"],
                        "metadata": {
                            "source": "isaacus/open-australian-legal-qa",
                            "question": item.get("question", ""),  # Store associated question
                            "answer": item.get("answer", "")  # Store associated answer
                        }
                    }
                    documents.append(doc)
                # If no context but has question and answer, create a document from those
                elif "question" in item and "answer" in item:
                    combined_text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
                    doc = {
                        "text": combined_text,
                        "metadata": {
                            "source": "isaacus/open-australian-legal-qa",
                            "is_qa_pair": True
                        }
                    }
                    documents.append(doc)

        elif self.dataset_name == "ibunescu/qa_legal_dataset_train":
            # This dataset has a different structure
            for item in tqdm(dataset["train"], desc="Processing documents"):
                # Adapt to the actual structure of this dataset
                if "text" in item and item["text"]:
                    doc = {
                        "text": item["text"],
                        "metadata": {
                            "source": "ibunescu/qa_legal_dataset_train",
                            "question": item.get("question", ""),
                            "answer": item.get("answer", "")
                        }
                    }
                    documents.append(doc)
                # If no text but has question and answer
                elif "question" in item and "answer" in item:
                    combined_text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
                    doc = {
                        "text": combined_text,
                        "metadata": {
                            "source": "ibunescu/qa_legal_dataset_train",
                            "is_qa_pair": True
                        }
                    }
                    documents.append(doc)

        else:
            # Generic approach for unknown dataset structure
            for item in tqdm(dataset["train"], desc="Processing documents"):
                # Try to identify the text field
                text_field = None
                for field in ["text", "context", "document", "content"]:
                    if field in item and item[field]:
                        text_field = field
                        break

                if text_field:
                    doc = {
                        "text": item[text_field],
                        "metadata": {
                            "source": self.dataset_name
                        }
                    }

                    # Add any other useful fields to metadata
                    for key, value in item.items():
                        if key != text_field and isinstance(value, (str, int, float, bool)):
                            doc["metadata"][key] = value

                    documents.append(doc)
                # If no text field found but has question and answer
                elif "question" in item and "answer" in item:
                    combined_text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
                    doc = {
                        "text": combined_text,
                        "metadata": {
                            "source": self.dataset_name,
                            "is_qa_pair": True
                        }
                    }
                    documents.append(doc)

        logger.info(f"Extracted {len(documents)} legal documents")
        return documents

    @staticmethod
    def process_documents(documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Process documents into chunks using legal chunking strategy."""
        logger.info("Processing documents into chunks")

        all_chunks = []

        for doc in tqdm(documents, desc="Chunking documents"):
            # Process each document using our specialized legal chunking
            try:
                chunks = process_legal_document(doc["text"], doc["metadata"])
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def save_processed_data(self, chunks: List[Chunk], output_dir: str = None):
        """Save processed chunks to disk."""
        if output_dir is None:
            output_dir = os.path.join(config["dataset"]["local_path"], "processed")

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving {len(chunks)} processed chunks to {output_dir}")

        # Save chunks as JSON
        output_file = os.path.join(output_dir, f"{self.dataset_name.replace('/', '_')}_chunks.json")

        with open(output_file, "w") as file:
            # Convert Chunk objects to dictionaries
            chunks_data = [
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "tokens": chunk.tokens
                }
                for chunk in chunks
            ]
            json.dump(chunks_data, file, indent=2)

        logger.info(f"Saved processed chunks to {output_file}")

    def run_ingestion_pipeline(self):
        """Run the full ingestion pipeline."""
        logger.info("Starting ingestion pipeline")

        # Step 1: Load dataset
        dataset = self.load_dataset()

        # Step 2: Extract documents
        documents = self.extract_legal_documents(dataset)

        # Step 3: Process documents into chunks
        chunks = self.process_documents(documents)

        # Step 4: Save processed data
        self.save_processed_data(chunks)

        logger.info("Completed ingestion pipeline")
        return chunks


if __name__ == "__main__":
    # Run ingestion directly if script is executed
    ingestion = LegalDataIngestion()
    ingestion.run_ingestion_pipeline()
