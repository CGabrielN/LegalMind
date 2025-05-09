#!/usr/bin/env python
"""
LegalMind Main Script

This is the main entry point for running the LegalMind system.
It provides a command-line interface for various operations.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root directory
project_root = Path(__file__).parent.parent


def setup_environment():
    """Set up the environment for running LegalMind."""
    # Add project root to path for imports
    sys.path.append(str(project_root))

    # Ensure required directories exist
    os.makedirs(project_root / "data" / "raw", exist_ok=True)
    os.makedirs(project_root / "data" / "processed", exist_ok=True)
    os.makedirs(project_root / "data" / "chroma_db", exist_ok=True)
    os.makedirs(project_root / "data" / "feedback", exist_ok=True)
    os.makedirs(project_root / "data" / "precedents", exist_ok=True)
    os.makedirs(project_root / "models", exist_ok=True)
    os.makedirs(project_root / "evaluation", exist_ok=True)

    logger.info("Environment setup complete")


def download_data():
    """Download the required datasets."""
    logger.info("Starting data download")
    script_path = project_root / "scripts" / "download_dataset.py"

    try:
        subprocess.run([sys.executable, script_path], check=True)
        logger.info("Data download completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data download failed: {str(e)}")
        sys.exit(1)


def process_data():
    """Process the downloaded data."""
    logger.info("Starting data processing")
    script_path = project_root / "scripts" / "process_data.py"

    try:
        subprocess.run([sys.executable, script_path], check=True)
        logger.info("Data processing completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data processing failed: {str(e)}")
        sys.exit(1)


def run_evaluation():
    """Run the evaluation on the system."""
    logger.info("Starting system evaluation")
    script_path = project_root / "scripts" / "run_evaluation.py"

    try:
        subprocess.run([sys.executable, script_path], check=True)
        logger.info("Evaluation completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


def run_ui():
    """Run the user interface."""
    logger.info("Starting LegalMind UI")
    ui_path = project_root / "src" / "ui" / "app.py"

    try:
        subprocess.run(["streamlit", "run", ui_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"UI execution failed: {str(e)}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Streamlit not found. Please install it with 'pip install streamlit'")
        sys.exit(1)


def initialize_rlhf():
    """Initialize the RLHF system with initial preference data."""
    logger.info("Initializing RLHF system")

    try:
        from src.models.hallucination_pipeline import LegalResponsePipeline

        # Initialize the pipeline
        pipeline = LegalResponsePipeline()
        pipeline.initialize_pipeline(seed_preferences=True)

        logger.info("RLHF system initialized successfully")
    except Exception as e:
        logger.error(f"RLHF initialization failed: {str(e)}")
        sys.exit(1)


def run_full_pipeline():
    """Run the full LegalMind pipeline."""
    logger.info("Starting full LegalMind pipeline")

    # Step 1: Download data
    download_data()

    # Step 2: Process data
    process_data()

    # Step 3: Initialize RLHF system
    initialize_rlhf()

    # Step 5: Start UI
    run_ui()


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="LegalMind - AI Legal Assistant")

    # Define command-line arguments
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--process", action="store_true", help="Process data")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--ui", action="store_true", help="Start the user interface")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--init-rlhf", action="store_true", help="Initialize RLHF system")

    args = parser.parse_args()

    # Set up environment
    setup_environment()

    # Execute requested operations
    if args.all:
        run_full_pipeline()
    else:
        if args.download:
            download_data()

        if args.process:
            process_data()

        if args.init_rlhf:
            initialize_rlhf()

        if args.evaluate:
            run_evaluation()

        if args.ui:
            run_ui()

        # If no operation specified, show help
        if not any([args.download, args.process, args.evaluate, args.ui,
                    args.init_rlhf, args.init_citations, args.test_hallucination,
                    args.train_rlhf, args.version]):
            parser.print_help()
            sys.exit(0)

    logger.info("LegalMind execution completed")


if __name__ == "__main__":
    main()
