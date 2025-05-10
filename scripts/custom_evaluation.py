#!/usr/bin/env python
"""
Custom LegalMind RAG Strategy Evaluation using LM Studio.
Split into two phases to avoid LM Studio API limitations.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import LegalMind modules
from src.core.resource_manager import ResourceManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomLegalEvaluator:
    """
    Evaluates different RAG strategies in LegalMind using LM Studio's Mistral model.
    Split into two phases to avoid API limitations.
    """

    def __init__(self, config_path=None, lm_studio_url=None):
        """Initialize the evaluator."""
        # Load configuration
        if config_path is None:
            config_path = project_root / "config" / "config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize resource manager
        self.resource_manager = ResourceManager(config_path)

        # Set LM Studio URL
        self.lm_studio_url = lm_studio_url or "http://127.0.0.1:1234/v1"

        # Connect to LLM for response generation
        self.llm = self.resource_manager.llm
        if self.llm is None:
            logger.error("Failed to initialize LLM. Ensure LM Studio is running.")
            sys.exit(1)

        # Initialize all RAG strategies
        self.basic_rag = self.resource_manager.basic_rag
        self.query_expansion = self.resource_manager.query_expander
        self.multi_query_rag = self.resource_manager.multi_query_rag
        self.metadata_enhanced_rag = self.resource_manager.metadata_enhanced_rag
        self.advanced_rag = self.resource_manager.advanced_rag

        # Create results directory
        self.results_dir = project_root / "evaluation" / "results"
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info("Custom LegalMind evaluator with LM Studio initialized")

    def load_test_queries(self, test_set_path=None):
        """
        Load test queries from a file or use defaults.
        """
        if test_set_path and Path(test_set_path).exists():
            with open(test_set_path, "r") as f:
                return json.load(f)

        # Default legal test queries
        return [
            "What constitutes negligence in New South Wales?",
            "Explain the duty of care concept in Australian tort law.",
            "What are the elements of a valid contract in Australian law?",
            "How does adverse possession work in Victoria?",
            "What rights do tenants have under Queensland rental laws?",
            "Explain how self-defense works as a legal defense in criminal cases.",
            "What is the statute of limitations for personal injury claims in Australia?",
            "How are damages calculated in breach of contract cases?",
            "What is the process for appealing a court decision to the High Court of Australia?",
            "Explain how property is divided in divorce cases in Australia.",
        ]

    def generate_response_with_strategy(self, query, strategy):
        """
        Generate a response using the specified RAG strategy.
        """
        logger.info(f"Generating response for query with {strategy} strategy: {query}")

        # Use the specified strategy to retrieve context
        if strategy == "basic":
            context, retrieved_docs = self.basic_rag.process_query(query)
        elif strategy == "query_expansion":
            expanded_queries = self.query_expansion.expand_query(query)
            expanded_query = expanded_queries[0] if expanded_queries else query
            context, retrieved_docs = self.basic_rag.process_query(expanded_query)
        elif strategy == "multi_query":
            context, retrieved_docs = self.multi_query_rag.process_query(query)
        elif strategy == "metadata_enhanced":
            context, retrieved_docs = self.metadata_enhanced_rag.process_query(query)
        elif strategy == "advanced":
            context, retrieved_docs = self.advanced_rag.process_query(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Generate response using LM Studio
        response = self.llm.generate(query, context)

        return response, context, retrieved_docs

    def evaluate_with_llm(self, query, response, context, criteria):
        """
        Use LLM to evaluate a response based on specific criteria.
        Severely truncates inputs to avoid context window issues.
        """
        # Truncate context and response to avoid context window issues
        truncated_context = context[:80] + "..." if len(context) > 80 else context
        truncated_response = response[:100] + "..." if len(response) > 100 else response

        # Define shorter evaluation prompts
        prompts = {
            "hallucination": f"""
                As a legal expert, evaluate this response for hallucinations:
                QUERY: {query}
                CONTEXT (EXCERPT): {truncated_context}
                RESPONSE: {truncated_response}
                Score hallucinations from 0.0 (none) to 1.0 (severe).
                SCORE: 
            """,

            "relevancy": f"""
                Legal expert: How relevant is this response to the query?
                QUERY: {query}
                RESPONSE: {truncated_response}
                Score relevancy from 0.0 (irrelevant) to 1.0 (perfectly relevant).
                SCORE: 
            """,

            "factual_consistency": f"""
                Legal expert: Is this response factually consistent with the context?
                QUERY: {query}
                CONTEXT (EXCERPT): {truncated_context}
                RESPONSE: {truncated_response}
                Score factual consistency from 0.0 (inconsistent) to 1.0 (perfectly consistent).
                SCORE: 
            """,
        }

        # Use the prompt for the specified criteria
        prompt = prompts.get(criteria)
        if not prompt:
            raise ValueError(f"Unknown evaluation criteria: {criteria}")

        # Make API call to LM Studio
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "mistral-7b-instruct-v0.2",
            "messages": [
                {"role": "system", "content": "You are a strict legal expert evaluating AI responses."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 128,  # Reduce token count significantly
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            evaluation_text = result["choices"][0]["message"]["content"]

            # Extract score with regex
            score_match = re.search(r"SCORE:\s*(\d+\.\d+)", evaluation_text)
            if score_match:
                score = float(score_match.group(1))
            else:
                score = 0.5  # Default score if parsing fails

            return {
                "score": score,
                "explanation": evaluation_text
            }

        except Exception as e:
            logger.error(f"Error evaluating with LLM: {str(e)}")
            return {
                "score": 0.5,
                "explanation": f"Evaluation error: {str(e)}"
            }

    @staticmethod
    def evaluate_with_heuristics(query, response, context):
        """
        Use simple heuristics for evaluation instead of LLM.
        """
        scores = {}

        # 1. Relevancy - simple keyword matching
        query_keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
        response_words = set(re.findall(r'\b\w{3,}\b', response.lower()))
        keyword_overlap = len(query_keywords.intersection(response_words)) / max(1, len(query_keywords))
        scores["relevancy"] = min(1.0, keyword_overlap * 2)  # Scale up a bit

        # 2. Citation count - number of references to Australian courts/cases
        citation_pattern = r'\[\d{4}\]\s+[A-Z]+'
        citation_count = len(re.findall(citation_pattern, response))
        scores["citation_quality"] = min(1.0, citation_count / 3)

        # 3. Structure - check if response has proper structure (bullet points or numbered lists)
        has_structure = bool(re.search(r'\d+\.|\*|\-', response))
        scores["structure"] = 0.8 if has_structure else 0.3

        # 4. Context usage - check if context terms appear in response
        if context:
            # Extract significant terms (capitalized words) from context
            context_important_terms = set(re.findall(r'\b[A-Z][a-z]{3,}\b', context))
            response_terms = set(re.findall(r'\b[A-Z][a-z]{3,}\b', response))
            context_term_usage = len(context_important_terms.intersection(response_terms)) / max(1,
                                                                                                 len(context_important_terms))
            scores["context_usage"] = min(1.0, context_term_usage * 2)
        else:
            scores["context_usage"] = 0.5

        # 5. Hallucination (inverse of context usage - higher if less context is used)
        scores["hallucination"] = 1.0 - scores["context_usage"] * 0.7

        # Average score (hallucination is considered inverse for average)
        non_hallucination_scores = [v for k, v in scores.items() if k != "hallucination"]
        avg_score = (sum(non_hallucination_scores) + (1.0 - scores["hallucination"])) / (len(scores))

        return scores, avg_score

    # ===== PHASE 1: GENERATE RESPONSES =====
    def generate_all_responses(self, queries, strategies, output_file=None):
        """
        Phase 1: Generate all responses for all queries and strategies and save to disk.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_file is None:
            output_file = self.results_dir / f"responses_{timestamp}.json"

        logger.info(f"Phase 1: Generating responses for {len(queries)} queries using {len(strategies)} strategies")

        results = {}
        for query_idx, query in enumerate(queries):
            logger.info(f"Processing query {query_idx + 1}/{len(queries)}: {query}")
            results[query] = {}

            for strategy in strategies:
                logger.info(f"Generating response with {strategy} strategy")

                try:
                    # Generate response
                    response, context, docs = self.generate_response_with_strategy(query, strategy)

                    # Store results
                    results[query][strategy] = {
                        "response": response,
                        "context": context[:500],  # Truncate context to save space
                        "num_docs": len(docs),
                        "timestamp": datetime.now().isoformat()
                    }

                    # Save after each response to protect against crashes
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)

                    logger.info(f"Response generated and saved for {strategy} strategy")

                    # Add delay to let LM Studio recover
                    time.sleep(5)

                except Exception as e:
                    logger.error(f"Error generating response for {strategy}: {str(e)}")
                    results[query][strategy] = {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }

        logger.info(f"Phase 1 complete. All responses saved to {output_file}")
        return output_file

    # ===== PHASE 2: EVALUATE RESPONSES =====
    def evaluate_saved_responses(self, responses_file, use_heuristics=True):
        """
        Phase 2: Evaluate previously generated responses.
        """
        logger.info(f"Phase 2: Evaluating responses from {responses_file}")

        # Load responses
        with open(responses_file, "r") as f:
            results = json.load(f)

        # Add evaluation results to each response
        for query in results:
            logger.info(f"Evaluating responses for query: {query}")

            for strategy in results[query]:
                # Skip if there was an error generating the response
                if "error" in results[query][strategy]:
                    logger.warning(f"Skipping evaluation for {strategy} due to generation error")
                    continue

                response = results[query][strategy]["response"]
                context = results[query][strategy]["context"]

                logger.info(f"Evaluating {strategy} strategy response")

                # Use heuristics for evaluation
                if use_heuristics:
                    scores, avg_score = self.evaluate_with_heuristics(query, response, context)
                    results[query][strategy]["evaluation"] = {
                        "method": "heuristics",
                        "scores": scores,
                        "average_score": avg_score
                    }
                    logger.info(f"Heuristic evaluation complete - Average score: {avg_score:.2f}")

                # Use LLM for evaluation (only if explicitly requested)
                else:
                    # Evaluate with each criterion
                    evaluation_results = {}
                    for criterion in ["hallucination", "relevancy", "factual_consistency"]:
                        logger.info(f"Evaluating {criterion}...")

                        try:
                            # Delay to let LM Studio recover
                            time.sleep(30)

                            # Get evaluation
                            evaluation = self.evaluate_with_llm(query, response, context, criterion)
                            score = evaluation.get("score", 0.5)

                            evaluation_results[criterion] = {
                                "score": score,
                                "explanation": evaluation.get("explanation", "")
                            }

                            logger.info(f"{criterion} score: {score:.2f}")
                        except Exception as e:
                            logger.error(f"Error evaluating {criterion}: {str(e)}")
                            evaluation_results[criterion] = {"score": 0.5, "error": str(e)}

                    # Calculate average score (invert hallucination score)
                    scores = [
                        1.0 - evaluation_results["hallucination"]["score"],  # Invert hallucination
                        evaluation_results["relevancy"]["score"],
                        evaluation_results["factual_consistency"]["score"]
                    ]
                    avg_score = sum(scores) / len(scores)

                    # Store evaluation results
                    results[query][strategy]["evaluation"] = {
                        "method": "llm",
                        "criteria": evaluation_results,
                        "average_score": avg_score
                    }

                    logger.info(f"LLM evaluation complete - Average score: {avg_score:.2f}")

        # Save evaluated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(responses_file).replace(".json", f"_evaluated_{timestamp}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Phase 2 complete. Evaluation results saved to {output_file}")
        return output_file, results

    # ===== PHASE 3: ANALYZE EVALUATION RESULTS =====
    def analyze_evaluation_results(self, results):
        """
        Analyze evaluation results to compare strategies.
        """
        logger.info("Phase 3: Analyzing evaluation results")

        # Extract all strategies
        strategies = set()
        for query in results:
            strategies.update(results[query].keys())
        strategies = sorted(strategies)

        # Create aggregated results
        aggregated = {strategy: {"scores": [], "query_results": {}} for strategy in strategies}

        for query in results:
            for strategy in strategies:
                if strategy in results[query] and "evaluation" in results[query][strategy]:
                    eval_data = results[query][strategy]["evaluation"]

                    # Get average score
                    avg_score = eval_data.get("average_score", 0.0)
                    aggregated[strategy]["scores"].append(avg_score)
                    aggregated[strategy]["query_results"][query] = avg_score

        # Calculate final averages
        for strategy in strategies:
            scores = aggregated[strategy]["scores"]
            if scores:
                aggregated[strategy]["average"] = sum(scores) / len(scores)
            else:
                aggregated[strategy]["average"] = 0.0

        # Sort strategies by average score
        sorted_strategies = sorted(
            [(s, aggregated[s]["average"]) for s in strategies],
            key=lambda x: x[1],
            reverse=True
        )

        # Create comparison summary
        summary = {
            "overall_ranking": [{"strategy": s, "average_score": score} for s, score in sorted_strategies],
            "strategy_details": aggregated,
            "queries_evaluated": list(results.keys()),
            "timestamp": datetime.now().isoformat()
        }

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"strategy_comparison_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Analysis complete. Summary saved to {summary_file}")
        return summary


def main():
    """Main function to run the evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LegalMind RAG strategies with LM Studio (Batch Method)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--test-set", help="Path to test queries JSON file")
    parser.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1", help="LM Studio API URL")
    parser.add_argument("--strategies", nargs="+", help="Strategies to evaluate")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to evaluate")
    parser.add_argument("--output-dir", help="Directory to save results")

    # Phase control arguments
    parser.add_argument("--phase", choices=["generate", "evaluate", "analyze", "all"],
                        default="all", help="Which phase to run")
    parser.add_argument("--responses-file", help="Path to previously generated responses JSON file")
    parser.add_argument("--use-heuristics", action="store_true", default=True,
                        help="Use heuristic evaluation instead of LLM")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = CustomLegalEvaluator(
        config_path=args.config,
        lm_studio_url=args.lm_studio_url
    )

    # Load test queries
    queries = evaluator.load_test_queries(args.test_set)

    # Limit number of queries if specified
    if args.num_queries and args.num_queries < len(queries):
        queries = queries[:args.num_queries]

    # Parse strategies to evaluate
    strategies = args.strategies
    if not strategies:
        strategies = ["basic", "query_expansion", "multi_query", "metadata_enhanced", "advanced"]

    # Execute requested phases
    responses_file = args.responses_file
    results = None

    if args.phase in ["generate", "all"]:
        responses_file = evaluator.generate_all_responses(queries, strategies)

    if args.phase in ["evaluate", "all"] and responses_file:
        eval_file, results = evaluator.evaluate_saved_responses(
            responses_file,
            use_heuristics=args.use_heuristics
        )

    if args.phase in ["analyze", "all"] and results:
        summary = evaluator.analyze_evaluation_results(results)

        # Print summary to console
        print("\n=== STRATEGY COMPARISON SUMMARY ===")
        for rank_info in summary["overall_ranking"]:
            strategy = rank_info["strategy"]
            score = rank_info["average_score"]
            print(f"{strategy.upper()}: {score:.3f}")

    elif args.phase == "analyze" and args.responses_file and "evaluated" in args.responses_file:
        # Load evaluation results if only analyzing
        with open(args.responses_file, "r") as f:
            results = json.load(f)
        summary = evaluator.analyze_evaluation_results(results)

        # Print summary to console
        print("\n=== STRATEGY COMPARISON SUMMARY ===")
        for rank_info in summary["overall_ranking"]:
            strategy = rank_info["strategy"]
            score = rank_info["average_score"]
            print(f"{strategy.upper()}: {score:.3f}")

    logger.info("Evaluation process complete")


if __name__ == "__main__":
    main()
