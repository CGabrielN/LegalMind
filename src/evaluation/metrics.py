"""
LegalMind Evaluation Metrics

This module provides metrics for evaluating the performance of the LegalMind
retrieval and generation system.
"""

import re
import yaml
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from ..embeddings.embedding import EmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class RetrievalEvaluator:
    """
    Evaluates the performance of the retrieval system.
    """

    def __init__(self):
        """Initialize the retrieval evaluator."""
        self.embedding_model = EmbeddingModel()
        logger.info("Initialized retrieval evaluator")

    def calculate_retrieval_precision(self, query: str, retrieved_docs: List[Dict[str, Any]], relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate precision of retrieval (proportion of retrieved documents that are relevant).

        Args:
            query: The query text
            retrieved_docs: List of retrieved documents
            relevant_docs: List of known relevant documents

        Returns:
            Precision score (0-1)
        """
        # Get IDs of relevant documents
        relevant_ids = set(doc.get("id", "") for doc in relevant_docs)

        # Count how many retrieved documents are relevant
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc.get("id", "") in relevant_ids)

        # Calculate precision
        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0

        logger.info(f"Retrieval precision: {precision:.4f}")
        return precision

    def calculate_retrieval_recall(self, retrieved_docs: List[Dict[str, Any]], relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate recall of retrieval (proportion of relevant documents that are retrieved).

        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of known relevant documents

        Returns:
            Recall score (0-1)
        """
        # Get IDs of retrieved documents
        retrieved_ids = set(doc.get("id", "") for doc in retrieved_docs)

        # Count how many relevant documents were retrieved
        relevant_retrieved = sum(1 for doc in relevant_docs if doc.get("id", "") in retrieved_ids)

        # Calculate recall
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0

        logger.info(f"Retrieval recall: {recall:.4f}")
        return recall

    def calculate_semantic_similarity(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate semantic similarity between query and retrieved documents.

        Args:
            query: The query text
            retrieved_docs: List of retrieved documents

        Returns:
            Average semantic similarity score (0-1)
        """
        if not retrieved_docs:
            return 0.0

        # Embed the query
        query_embedding = self.embedding_model.embed_text(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Embed each document
        doc_embeddings = []
        for doc in retrieved_docs:
            doc_text = doc["text"]
            doc_embedding = self.embedding_model.embed_text(doc_text)
            doc_embeddings.append(doc_embedding)

        # Convert to numpy array
        doc_embeddings = np.array(doc_embeddings)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Average similarity
        avg_similarity = np.mean(similarities)

        logger.info(f"Average semantic similarity: {avg_similarity:.4f}")
        return float(avg_similarity)

    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]], relevant_docs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Evaluate retrieval performance with multiple metrics.

        Args:
            query: The query text
            retrieved_docs: List of retrieved documents
            relevant_docs: Optional list of known relevant documents

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        # If we have ground truth relevant documents, calculate precision and recall
        if relevant_docs:
            results["precision"] = self.calculate_retrieval_precision(query, retrieved_docs, relevant_docs)
            results["recall"] = self.calculate_retrieval_recall(retrieved_docs, relevant_docs)

            # Calculate F1 score
            if results["precision"] + results["recall"] > 0:
                results["f1"] = 2 * results["precision"] * results["recall"] / (results["precision"] + results["recall"])
            else:
                results["f1"] = 0.0

        # Calculate semantic similarity
        results["semantic_similarity"] = self.calculate_semantic_similarity(query, retrieved_docs)

        return results


class ResponseEvaluator:
    """
    Evaluates the quality of generated legal responses.
    """

    def __init__(self):
        """Initialize the response evaluator."""
        self.embedding_model = EmbeddingModel()
        logger.info("Initialized response evaluator")

    def count_citations(self, text: str) -> int:
        """
        Count legal citations in the text.

        Args:
            text: The text to analyze

        Returns:
            Number of legal citations
        """
        # Pattern for Australian citations
        pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"

        # Find all citations
        citations = re.findall(pattern, text)

        return len(citations)

    def calculate_citation_accuracy(self, response: str, reference_citations: List[str]) -> float:
        """
        Calculate accuracy of citations in the response.

        Args:
            response: Generated response
            reference_citations: List of correct citations

        Returns:
            Citation accuracy score (0-1)
        """
        if not reference_citations:
            return 1.0 if self.count_citations(response) == 0 else 0.0

        # Extract citations from response
        pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"
        response_citations = re.findall(pattern, response)

        # Count correct citations
        correct_citations = sum(1 for citation in response_citations if citation in reference_citations)

        # Calculate precision
        precision = correct_citations / len(response_citations) if response_citations else 0.0

        # Calculate recall
        recall = correct_citations / len(reference_citations) if reference_citations else 0.0

        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        logger.info(f"Citation accuracy (F1): {f1:.4f}")
        return f1

    def evaluate_response_completeness(self, response: str, reference_answer: str) -> float:
        """
        Evaluate completeness of the response compared to reference answer.

        Args:
            response: Generated response
            reference_answer: Reference answer

        Returns:
            Completeness score (0-1)
        """
        # Embed both texts
        response_embedding = self.embedding_model.embed_text(response)
        reference_embedding = self.embedding_model.embed_text(reference_answer)

        # Convert to numpy arrays
        response_embedding = np.array(response_embedding).reshape(1, -1)
        reference_embedding = np.array(reference_embedding).reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(response_embedding, reference_embedding)[0][0]

        logger.info(f"Response completeness: {similarity:.4f}")
        return float(similarity)

    def check_for_hallucinations(self, response: str, context: str) -> Tuple[bool, List[str]]:
        """
        Check for potential hallucinations in the response.

        Args:
            response: Generated response
            context: Context provided to the model

        Returns:
            Tuple of (has_hallucinations, list_of_suspicious_statements)
        """
        # Extract citations from response
        pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"
        response_citations = re.findall(pattern, response)

        # Check if citations are in the context
        suspicious_citations = []
        for citation in response_citations:
            if citation not in context:
                suspicious_citations.append(f"Citation not found in context: {citation}")

        # Check for definitive statements without support
        definitive_patterns = [
            r"always requires",
            r"never allows",
            r"all cases",
            r"in every situation",
            r"without exception"
        ]

        suspicious_statements = []
        for pattern in definitive_patterns:
            matches = re.findall(pattern, response.lower())
            for match in matches:
                if match not in context.lower():
                    suspicious_statements.append(f"Definitive statement without support: '{match}'")

        has_hallucinations = len(suspicious_citations) > 0 or len(suspicious_statements) > 0
        all_suspicious = suspicious_citations + suspicious_statements

        return has_hallucinations, all_suspicious

    def evaluate_response(self, response: str, reference_answer: Optional[str] = None, context: Optional[str] = None, reference_citations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the generated response with multiple metrics.

        Args:
            response: Generated response
            reference_answer: Optional reference answer for comparison
            context: Optional context provided to the model
            reference_citations: Optional list of correct citations

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        # Count citations
        results["citation_count"] = self.count_citations(response)

        # If we have reference citations, calculate citation accuracy
        if reference_citations:
            results["citation_accuracy"] = self.calculate_citation_accuracy(response, reference_citations)

        # If we have a reference answer, evaluate completeness
        if reference_answer:
            results["completeness"] = self.evaluate_response_completeness(response, reference_answer)

        # If we have context, check for hallucinations
        if context:
            has_hallucinations, suspicious = self.check_for_hallucinations(response, context)
            results["has_hallucinations"] = has_hallucinations
            results["suspicious_statements"] = suspicious

        return results


class LegalMindEvaluator:
    """
    Complete evaluator for the LegalMind system.
    """

    def __init__(self):
        """Initialize the complete evaluator."""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.response_evaluator = ResponseEvaluator()
        logger.info("Initialized LegalMind evaluator")

    def evaluate_full_pipeline(self, query: str, retrieved_docs: List[Dict[str, Any]], response: str, context: str, reference_answer: Optional[str] = None, relevant_docs: Optional[List[Dict[str, Any]]] = None, reference_citations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the full LegalMind pipeline.

        Args:
            query: The query text
            retrieved_docs: List of retrieved documents
            response: Generated response
            context: Context provided to the model
            reference_answer: Optional reference answer
            relevant_docs: Optional list of relevant documents
            reference_citations: Optional list of correct citations

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        # Evaluate retrieval
        retrieval_results = self.retrieval_evaluator.evaluate_retrieval(query, retrieved_docs, relevant_docs)
        results["retrieval"] = retrieval_results

        # Evaluate response
        response_results = self.response_evaluator.evaluate_response(response, reference_answer, context, reference_citations)
        results["response"] = response_results

        # Calculate overall score
        overall_score = 0.0
        components = 0

        # Add retrieval components
        if "f1" in retrieval_results:
            overall_score += retrieval_results["f1"]
            components += 1

        overall_score += retrieval_results["semantic_similarity"]
        components += 1

        # Add response components
        if "citation_accuracy" in response_results:
            overall_score += response_results["citation_accuracy"]
            components += 1

        if "completeness" in response_results:
            overall_score += response_results["completeness"]
            components += 1

        if "has_hallucinations" in response_results:
            # Add 1.0 if no hallucinations, 0.0 if hallucinations
            overall_score += 0.0 if response_results["has_hallucinations"] else 1.0
            components += 1

        # Calculate average
        if components > 0:
            results["overall_score"] = overall_score / components
        else:
            results["overall_score"] = 0.0

        logger.info(f"Overall evaluation score: {results['overall_score']:.4f}")
        return results