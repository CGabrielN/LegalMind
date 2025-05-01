#!/usr/bin/env python
"""
LegalMind Main Script

This is the main entry point for running the LegalMind system.
It provides a command-line interface for various operations.
"""

import os
import sys
import argparse
import logging
import subprocess
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

def initialize_citation_database():
    """Initialize the citation database with Australian precedents."""
    logger.info("Initializing citation database")

    try:
        from src.models.citation_handler import CitationCrossReferencer, initialize_australian_precedents

        # Initialize with pre-defined precedents
        precedents = initialize_australian_precedents()

        # Create cross-referencer and save precedents
        cross_referencer = CitationCrossReferencer()
        cross_referencer.precedents = precedents
        cross_referencer.save_precedents()

        logger.info("Citation database initialized successfully")
    except Exception as e:
        logger.error(f"Citation database initialization failed: {str(e)}")
        sys.exit(1)

def run_hallucination_tests():
    """Run hallucination detection and mitigation tests."""
    logger.info("Running hallucination mitigation tests")

    try:
        import unittest
        from tests.test_hallucination_mitigation import (
            TestHallucinationDetector,
            TestHallucinationMitigation,
            TestCitationVerification
        )

        # Create test suite
        suite = unittest.TestSuite()
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestHallucinationDetector))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestHallucinationMitigation))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCitationVerification))

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            logger.info("Hallucination mitigation tests completed successfully")
        else:
            logger.error("Hallucination mitigation tests failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Hallucination tests failed: {str(e)}")
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

    # Step 4: Initialize citation database
    initialize_citation_database()

    # Step 5: Run hallucination tests
    run_hallucination_tests()

    # Step 6: Run evaluation
    run_evaluation()

    # Step 7: Start UI
    run_ui()

def run_weekly_training():
    """Run weekly RLHF training from collected feedback."""
    logger.info("Running weekly RLHF training")

    try:
        from src.models.hallucination_pipeline import LegalResponsePipeline

        # Initialize pipeline
        pipeline = LegalResponsePipeline()

        # Train from feedback
        pipeline.train_from_feedback()

        logger.info("Weekly RLHF training completed successfully")
    except Exception as e:
        logger.error(f"Weekly RLHF training failed: {str(e)}")
        sys.exit(1)

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
    parser.add_argument("--init-citations", action="store_true", help="Initialize citation database")
    parser.add_argument("--test-hallucination", action="store_true", help="Run hallucination tests")
    parser.add_argument("--train-rlhf", action="store_true", help="Run weekly RLHF training")
    parser.add_argument("--version", action="store_true", help="Display LegalMind version info")

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

        if args.init_citations:
            initialize_citation_database()

        if args.test_hallucination:
            run_hallucination_tests()

        if args.evaluate:
            run_evaluation()

        if args.ui:
            run_ui()

        if args.train_rlhf:
            run_weekly_training()

        if args.version:
            try:
                from src.models.hallucination_pipeline import LegalResponsePipeline
                pipeline = LegalResponsePipeline()
                version_info = pipeline.get_version_info()
                print("\nLegalMind System Version Information:")
                for key, value in version_info.items():
                    print(f"  {key}: {value}")
                print("\n")
            except Exception as e:
                logger.error(f"Error displaying version info: {str(e)}")

        # If no operation specified, show help
        if not any([args.download, args.process, args.evaluate, args.ui,
                    args.init_rlhf, args.init_citations, args.test_hallucination,
                    args.train_rlhf, args.version]):
            parser.print_help()
            sys.exit(0)

    logger.info("LegalMind execution completed")

if __name__ == "__main__":
    main()