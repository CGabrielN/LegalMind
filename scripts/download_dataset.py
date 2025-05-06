#!/usr/bin/env python
"""
Download datasets for LegalMind.

This script downloads the required datasets from Hugging Face
and saves them locally.
"""

import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root to path to import project modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.ingestion import LegalDataIngestion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to download and save datasets.
    """
    logger.info("Starting dataset download")

    # Load config
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create data directories
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Try primary dataset
    try:
        logger.info(f"Downloading primary dataset: {config['dataset']['name']}")
        ingestion = LegalDataIngestion(dataset_name=config["dataset"]["name"])
        dataset = ingestion.load_dataset()
        logger.info(f"Successfully downloaded dataset with {len(dataset['train'])} training examples")

        # Save dataset info
        dataset_info = {
            "name": config["dataset"]["name"],
            "size": {split: len(dataset[split]) for split in dataset},
            "features": list(dataset["train"][0].keys()) if len(dataset["train"]) > 0 else []
        }

        with open(raw_dir / "dataset_info.yaml", "w") as f:
            yaml.dump(dataset_info, f)

        logger.info(f"Saved dataset info to {raw_dir / 'dataset_info.yaml'}")

    except Exception as e:
        logger.error(f"Error downloading primary dataset: {str(e)}")

        # Try alternative dataset
        try:
            logger.info(f"Downloading alternative dataset: {config['dataset']['alternative']}")
            ingestion = LegalDataIngestion(dataset_name=config["dataset"]["alternative"])
            dataset = ingestion.load_dataset()
            logger.info(f"Successfully downloaded alternative dataset with {len(dataset['train'])} training examples")

            # Save dataset info
            dataset_info = {
                "name": config["dataset"]["alternative"],
                "size": {split: len(dataset[split]) for split in dataset},
                "features": list(dataset["train"][0].keys()) if len(dataset["train"]) > 0 else []
            }

            with open(raw_dir / "dataset_info.yaml", "w") as f:
                yaml.dump(dataset_info, f)

            logger.info(f"Saved dataset info to {raw_dir / 'dataset_info.yaml'}")

        except Exception as e2:
            logger.error(f"Error downloading alternative dataset: {str(e2)}")
            logger.error("Failed to download any dataset")
            return

    logger.info("Dataset download complete")


if __name__ == "__main__":
    main()
