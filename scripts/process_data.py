#!/usr/bin/env python
"""
Process datasets for LegalMind.

This script processes the downloaded datasets, applies chunking strategies,
and stores the processed data in the vector database.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add project root to path to import project modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.ingestion import LegalDataIngestion
from src.embeddings.embedding import EmbeddingModel
from src.vectordb.chroma_db import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to process datasets and load them into the vector database.
    """
    logger.info("Starting data processing pipeline")

    # Load config
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Step 1: Initialize components
    logger.info("Initializing components")
    try:
        ingestion = LegalDataIngestion()
    except Exception as e:
        logger.error(f"Error initializing data ingestion: {str(e)}")
        return

    try:
        embedding_model = EmbeddingModel(config)
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        logger.warning("Continuing without embedding model. Some functionality will be limited.")
        embedding_model = None

    try:
        vector_store = ChromaVectorStore()
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return

    # Step 2: Load and process dataset
    logger.info("Loading and processing dataset")
    try:
        # Load dataset
        dataset = ingestion.load_dataset()

        # Extract documents
        documents = ingestion.extract_legal_documents(dataset)
        logger.info(f"Extracted {len(documents)} documents")

        # Process documents into chunks
        chunks = ingestion.process_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Save processed chunks
        processed_dir = project_root / "data" / "processed"
        os.makedirs(processed_dir, exist_ok=True)
        ingestion.save_processed_data(chunks, output_dir=str(processed_dir))

    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        return

    # Step 3: Embed chunks if embedding model is available
    if embedding_model:
        logger.info("Embedding chunks")
        try:
            # Embed chunks
            embedded_chunks = embedding_model.embed_chunks(chunks)
            logger.info(f"Embedded {len(embedded_chunks)} chunks")

        except Exception as e:
            logger.error(f"Error in embedding: {str(e)}")
            logger.warning("Continuing with non-embedded chunks")
            embedded_chunks = chunks
    else:
        logger.warning("Skipping embedding step - embedding model not available")
        embedded_chunks = chunks

    # Step 4: Load into vector database
    logger.info("Loading chunks into vector database")
    try:
        # Reset vector store to start fresh
        vector_store.reset()

        # Add chunks to vector store
        vector_store.add_chunks(embedded_chunks)

        # Verify document count
        doc_count = vector_store.count_documents()
        logger.info(f"Successfully loaded {doc_count} documents into vector database")

    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        return

    logger.info("Data processing pipeline complete")

if __name__ == "__main__":
    main()