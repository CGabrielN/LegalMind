"""
LegalMind Vector Database

This module handles the Chroma vector database integration for storing
and retrieving legal document embeddings.
"""

import os
import yaml
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class ChromaVectorStore:
    """Manages the Chroma vector database for legal document retrieval."""

    def __init__(self):
        """Initialize the Chroma client and collection."""
        self.persist_directory = config["vectordb"]["persist_directory"]
        self.collection_name = config["vectordb"]["collection_name"]

        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True
            )
        )

        # Get or create collection
        self._initialize_collection()

        logger.info(f"Initialized Chroma vector store at {self.persist_directory}")

    def _initialize_collection(self):
        """Initialize or get the Chroma collection."""
        # Use the specified embedding model
        embedding_model = config["embedding"]["model_name"]

        # Setup embedding function based on model
        try:
            # First try the default sentence transformer embedding function
            # This doesn't require an API key
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            logger.info(f"Using SentenceTransformerEmbeddingFunction with model: {embedding_model}")
        except Exception as e:
            logger.warning(f"Error initializing SentenceTransformerEmbeddingFunction: {str(e)}")
            logger.info("Using default embedding function")
            # Fall back to default embedding function (no model required)
            self.ef = None

        # Get or create collection with the embedding function
        try:
            if self.ef:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.ef
                )
            else:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            if self.ef:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.ef,
                    metadata={"description": "Australian legal documents collection"}
                )
            else:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Australian legal documents collection"}
                )

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Add documents to the vector store.

        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            batch_size: Number of documents to process at once
        """
        logger.info(f"Adding {len(documents)} documents to Chroma")

        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            # Prepare batch data
            ids = [str(uuid.uuid4()) for _ in range(len(batch))]
            texts = [doc["text"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]

            # Add to collection
            try:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error adding batch to Chroma: {str(e)}")

        logger.info(f"Completed adding documents to Chroma")

    def add_chunks(self, chunks: List[Any], batch_size: int = 100):
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to process at once
        """
        # Convert chunks to the format expected by add_documents
        documents = []
        for chunk in chunks:
            # Handle chunks with or without 'embedding' in metadata
            metadata_copy = chunk.metadata.copy()
            if 'embedding' in metadata_copy:
                # We don't want to store the embedding in metadata
                # ChromaDB will handle this separately
                del metadata_copy['embedding']

            documents.append({
                "text": chunk.text,
                "metadata": metadata_copy
            })

        self.add_documents(documents, batch_size)

    def query(self,
              query_text: str,
              n_results: int = 5,
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: The query text
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            Dictionary with query results
        """
        logger.info(f"Querying vector store with: '{query_text[:50]}...'")

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                where_document=where_document
            )

            logger.info(f"Found {len(results['documents'][0])} matching documents")
            return results
        except Exception as e:
            logger.error(f"Error querying Chroma: {str(e)}")
            # Return empty results
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

    def metadata_filter_query(self, query_text: str,
                              filters: Dict[str, Union[str, List[str]]],
                              n_results: int = 5) -> Dict[str, Any]:
        """
        Query with metadata filters for jurisdiction, document type, etc.

        Args:
            query_text: The query text
            filters: Dictionary of metadata filters
            n_results: Number of results to return

        Returns:
            Dictionary with query results
        """
        # Construct where clauses
        where_clause = {}

        for key, value in filters.items():
            if isinstance(value, list):
                # For list values, we want any match (OR)
                where_clause[key] = {"$in": value}
            else:
                # For single values, exact match
                where_clause[key] = value

        return self.query(query_text, n_results, where=where_clause)

    def count_documents(self) -> int:
        """Get the count of documents in the collection."""
        return self.collection.count()

    def reset(self):
        """Reset the collection, removing all documents."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self._initialize_collection()