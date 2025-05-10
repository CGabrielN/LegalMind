"""
LegalMind Embedding Module

This module handles the embedding of legal texts using the specified model.
"""

import logging
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Handles the embedding of legal texts using the BGE model."""

    def __init__(self, config):
        """Initialize the embedding model from config.
        Args:
            config (dict): Configuration object containing the model configuration.
        """
        self.config = config
        self.model_name = self.config["embedding"]["model_name"]
        self.device = self.config["embedding"]["device"]
        self.batch_size = self.config["embedding"]["batch_size"]
        self.max_length = self.config["embedding"]["max_length"]

        # Check if CUDA is available when device is set to cuda
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"Loading embedding model: {self.model_name}")
        self._load_model()

    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            logger.info(f"Successfully loaded model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]

        # Expand attention mask to same dimensions as token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum token embeddings and divide by the total number of tokens
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        # Prepare the text for the model
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move tensors to the correct device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Compute token embeddings with no gradient computation
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list and return
        return embeddings[0].cpu().tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i + self.batch_size]

            # Prepare the batch for the model
            batch_embeddings = self.embed_text(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document dictionaries.

        Args:
            documents: List of document dictionaries with 'text' field

        Returns:
            List of document dictionaries with added 'embedding' field
        """
        texts = [doc["text"] for doc in documents]
        embeddings = self.embed_texts(texts)

        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i]

        return documents

    def embed_chunks(self, chunks: List[Any]) -> List[Any]:
        """
        Embed a list of Chunk objects.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of Chunk objects with embeddings added to their metadata
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        # Add embeddings to chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["embedding"] = embeddings[i]

        return chunks
