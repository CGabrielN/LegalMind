#!/usr/bin/env python
"""
Inspect the structure of HuggingFace datasets for LegalMind.

This script helps debug dataset loading by showing the structure and contents
of the datasets.
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from pprint import pprint
from datasets import load_dataset

# Add project root to path to import project modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_dataset(dataset_name: str, num_samples: int = 3):
    """
    Inspect the structure and content of a HuggingFace dataset.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        num_samples: Number of samples to show
    """
    logger.info(f"Loading dataset: {dataset_name}")

    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)

        # Get available splits
        splits = list(dataset.keys())
        logger.info(f"Available splits: {splits}")

        # Inspect each split
        for split in splits:
            logger.info(f"\nInspecting split: {split}")
            split_data = dataset[split]

            # Get dataset size
            size = len(split_data)
            logger.info(f"Number of examples: {size}")

            # Get features/columns
            features = list(split_data.features.keys())
            logger.info(f"Features: {features}")

            # Show sample examples
            logger.info(f"\nSample examples (showing {min(num_samples, size)}):")
            for i in range(min(num_samples, size)):
                example = split_data[i]
                print(f"\nExample {i+1}:")

                # Print each field
                for feature in features:
                    value = example[feature]
                    # Truncate long text values
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    print(f"  {feature}: {value}")

            # Check for any empty fields
            empty_fields = {}
            for feature in features:
                empty_count = sum(1 for i in range(size) if not split_data[i][feature])
                if empty_count > 0:
                    empty_fields[feature] = empty_count

            if empty_fields:
                logger.info("\nEmpty field counts:")
                for field, count in empty_fields.items():
                    logger.info(f"  {field}: {count} empty values ({count/size*100:.1f}%)")

    except Exception as e:
        logger.error(f"Error inspecting dataset: {str(e)}")

def main():
    """Main function to inspect datasets."""
    # Load config
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get dataset names from config
    primary_dataset = config["dataset"]["name"]
    alternative_dataset = config["dataset"].get("alternative")

    # Inspect primary dataset
    logger.info(f"Inspecting primary dataset: {primary_dataset}")
    inspect_dataset(primary_dataset)

    # Inspect alternative dataset if provided
    if alternative_dataset:
        logger.info(f"\nInspecting alternative dataset: {alternative_dataset}")
        inspect_dataset(alternative_dataset)

if __name__ == "__main__":
    main()