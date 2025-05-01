#!/usr/bin/env python
"""
Evaluate LegalMind performance.

This script runs evaluation on the LegalMind system using a set of test queries
and produces performance metrics.
"""

import os
import sys
import yaml
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path to import project modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.retrieval.basic_rag import BasicRAG
from src.models.llm import LegalLLM
from src.evaluation.metrics import LegalMindEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test queries for evaluation
TEST_QUERIES = [
    "What are the elements of negligence in Australian law?",
    "Explain the legal principle of duty of care in New South Wales.",
    "What constitutes a breach of contract in Australian law?",
    "How does adverse possession work in Victoria?",
    "What are the requirements for a valid will in Queensland?",
    "Explain the concept of statutory interpretation in Australian law.",
    "What is the role of precedent in the Australian legal system?",
    "How does the Consumer Law protect customers in Australia?",
    "What are the key principles of family law in Australia?",
    "Explain the legal process for property settlements in divorce cases."
]

def main():
    """
    Main function to run evaluation on the LegalMind system.
    """
    logger.info("Starting LegalMind evaluation")

    # Load config
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create output directory
    eval_dir = project_root / "evaluation"
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize components
    logger.info("Initializing evaluation components")
    rag = BasicRAG()
    llm = LegalLLM()
    evaluator = LegalMindEvaluator()

    # Results storage
    all_results = []

    # Process test queries
    logger.info(f"Processing {len(TEST_QUERIES)} test queries")
    for i, query in enumerate(TEST_QUERIES):
        logger.info(f"Evaluating query {i+1}/{len(TEST_QUERIES)}: {query[:50]}...")

        try:
            # Retrieve documents
            context, retrieved_docs = rag.process_query(query)

            # Generate response
            response = llm.generate(query, context)

            # Evaluate
            eval_results = evaluator.evaluate_full_pipeline(
                query=query,
                retrieved_docs=retrieved_docs,
                response=response,
                context=context
            )

            # Store results
            result = {
                "query_id": i + 1,
                "query": query,
                "response": response,
                "num_docs_retrieved": len(retrieved_docs),
                "retrieval_semantic_similarity": eval_results["retrieval"]["semantic_similarity"],
                "citation_count": eval_results["response"]["citation_count"],
                "has_hallucinations": eval_results["response"].get("has_hallucinations", False),
                "overall_score": eval_results["overall_score"]
            }

            all_results.append(result)

            # Save individual result
            with open(eval_dir / f"query_{i+1}_results.json", "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"Completed evaluation for query {i+1} with score: {eval_results['overall_score']:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating query {i+1}: {str(e)}")

    # Calculate aggregate metrics
    if all_results:
        # Convert to dataframe for analysis
        df = pd.DataFrame(all_results)

        # Calculate summary statistics
        summary = {
            "num_queries": len(df),
            "avg_overall_score": df["overall_score"].mean(),
            "avg_semantic_similarity": df["retrieval_semantic_similarity"].mean(),
            "avg_citation_count": df["citation_count"].mean(),
            "hallucination_rate": df["has_hallucinations"].mean() if "has_hallucinations" in df.columns else None,
            "best_query_id": df.loc[df["overall_score"].idxmax(), "query_id"],
            "best_query_score": df["overall_score"].max(),
            "worst_query_id": df.loc[df["overall_score"].idxmin(), "query_id"],
            "worst_query_score": df["overall_score"].min()
        }

        # Save summary
        with open(eval_dir / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save full results
        df.to_csv(eval_dir / "evaluation_results.csv", index=False)

        logger.info(f"Evaluation complete. Average score: {summary['avg_overall_score']:.4f}")
        logger.info(f"Results saved to {eval_dir}")
    else:
        logger.error("No evaluation results were generated")

if __name__ == "__main__":
    main()