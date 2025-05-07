"""
LegalMind Embedding Module using LM Studio API

This module handles the embedding of legal texts using LM Studio's API
"""

import json
import logging
from typing import List, Dict, Any

import numpy as np
import requests
import yaml
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class EmbeddingModel:
    """Handles the embedding of legal texts using LM Studio API."""

    _instance = None  # Singleton instance

    def __new__(cls):
        """Singleton pattern to ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the embedding model from config."""
        # Skip initialization if already initialized (singleton pattern)
        if getattr(self, '_initialized', False):
            return

        self.model_name = config["embedding"]["model_name"]
        self.batch_size = config["embedding"]["batch_size"]
        self.max_length = config["embedding"]["max_length"]
        self.embedding_dim = 1024  # Default embedding dimension for BGE models

        # Get LM Studio API URL from config or use default
        self.api_base_url = config.get("lm_studio", {}).get("api_base_url", "http://127.0.0.1:1234/v1")
        self.embeddings_url = f"{self.api_base_url}/embeddings"

        # Test connection to LM Studio API
        self._test_connection()

        logger.info(f"Initialized embedding model via LM Studio API at {self.api_base_url}")
        self._initialized = True

    def _test_connection(self):
        """Test the connection to the LM Studio API."""
        try:
            response = requests.get(self.api_base_url)
            if response.status_code == 404:
                # This is actually expected - the root endpoint doesn't exist
                # but tells us the server is running
                logger.info("LM Studio API is reachable")
            else:
                logger.info(f"LM Studio API responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to LM Studio API at {self.api_base_url}")
            logger.error("Please ensure LM Studio is running and the API is enabled.")
            raise ConnectionError(f"Failed to connect to LM Studio API at {self.api_base_url}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string using LM Studio API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        # Prepare the API request
        payload = {
            "model": self.model_name,
            "input": text
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Make the API call
            response = requests.post(
                self.embeddings_url,
                headers=headers,
                data=json.dumps(payload)
            )

            # Check for errors
            response.raise_for_status()

            # Parse the response
            result = response.json()

            if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                embedding = result["data"][0]["embedding"]
                return np.array(embedding)
            else:
                logger.error(f"Unexpected API response structure: {result}")
                # Return a zero vector of an appropriate size in case of error
                return np.zeros(self.embedding_dim)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LM Studio API for embeddings: {str(e)}")
            # Return a zero vector in case of error
            return np.zeros(self.embedding_dim)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts in batches using LM Studio API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors as numpy arrays
        """
        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = []

            # Create a single request for the batch if supported
            try:
                # Prepare the API request for the batch
                payload = {
                    "model": self.model_name,
                    "input": batch_texts
                }

                headers = {
                    "Content-Type": "application/json"
                }

                # Make the API call
                response = requests.post(
                    self.embeddings_url,
                    headers=headers,
                    data=json.dumps(payload)
                )

                # Check for errors
                response.raise_for_status()

                # Parse the response
                result = response.json()

                if "data" in result and len(result["data"]) == len(batch_texts):
                    for item in result["data"]:
                        if "embedding" in item:
                            batch_embeddings.append(np.array(item["embedding"]))
                        else:
                            # Fallback for any items missing embeddings
                            logger.warning("Missing embedding in API response")
                            batch_embeddings.append(np.zeros(self.embedding_dim))
                else:
                    # Fallback to individual requests
                    raise ValueError("Invalid API response structure")

            except Exception as e:
                logger.warning(f"Batch embedding failed, falling back to individual requests: {str(e)}")
                # Fall back to individual requests
                for text in batch_texts:
                    embedding = self.embed_text(text)
                    batch_embeddings.append(embedding)

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
