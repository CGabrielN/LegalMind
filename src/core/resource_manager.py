"""
LegalMind Resource Manager

This module provides centralized management of shared resources
to avoid duplication and improve performance.
"""

import logging
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages shared resources for the LegalMind application.

    This class implements the Singleton pattern to ensure only one
    instance of each resource exists across the application.
    """

    _instance = None

    def __new__(cls, config_path=None):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        """Initialize shared resources."""
        # Skip initialization if already done
        if self._initialized:
            return

        # Set default config path if not provided
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        logger.info(f"Initializing ResourceManager with config: {config_path}")

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize placeholders for lazy-loaded resources
        self._vector_store = None
        self._embedding_model = None
        self._llm = None
        self._citation_verifier = None
        self._citation_formatter = None
        self._hallucination_mitigation = None
        self._legal_response_pipeline = None

        self._query_expander = None
        self._basic_rag = None
        self._multi_query_rag = None
        self._metadata_enhanced_rag = None
        self._advanced_rag = None
        self._initialized = True
        logger.info("ResourceManager initialized")

    @property
    def vector_store(self):
        """Lazy-loaded vector store."""
        if self._vector_store is None:
            from src.vectordb.chroma_db import ChromaVectorStore
            logger.info("Initializing shared ChromaVectorStore")
            self._vector_store = ChromaVectorStore(self.config)
        return self._vector_store

    @property
    def embedding_model(self):
        """Lazy-loaded embedding model."""
        if self._embedding_model is None:
            from src.embeddings.embedding import EmbeddingModel
            logger.info("Initializing shared EmbeddingModel")
            self._embedding_model = EmbeddingModel(self.config)
        return self._embedding_model

    @property
    def query_expander(self):
        """Lazy-loaded query expander."""
        if self._query_expander is None:
            from src.retrieval.query_expansion import LegalQueryExpansion
            logger.info("Initializing shared LegalQueryExpansion")
            self._query_expander = LegalQueryExpansion()
        return self._query_expander

    @property
    def llm(self):
        """Lazy-loaded LLM API client."""
        if self._llm is None:
            from src.models.llm_api import LMStudioAPI
            try:
                lm_studio_url = self.config.get("lm_studio", {}).get("api_base_url", "http://127.0.0.1:1234/v1")
                logger.info(f"Initializing shared LMStudioAPI at {lm_studio_url}")
                self._llm = LMStudioAPI(api_base_url=lm_studio_url, config=self.config)
            except Exception as e:
                logger.error(f"Error connecting to LM Studio API: {str(e)}")
                self._llm = None
        return self._llm

    @property
    def citation_verifier(self):
        """Lazy-loaded citation verifier."""
        if self._citation_verifier is None:
            from src.models.citation_handler import CitationVerification
            logger.info("Initializing shared CitationVerification")
            self._citation_verifier = CitationVerification()
        return self._citation_verifier

    @property
    def citation_formatter(self):
        """Lazy-loaded citation formatter."""
        if self._citation_formatter is None:
            from src.models.citation_handler import CitationFormatting
            logger.info("Initializing shared CitationFormatting")
            self._citation_formatter = CitationFormatting()
        return self._citation_formatter

    @property
    def hallucination_mitigation(self):
        """Lazy-loaded hallucination mitigation."""
        if self._hallucination_mitigation is None:
            from src.models.hallucination_detector import HallucinationMitigation
            logger.info("Initializing shared HallucinationMitigation")
            self._hallucination_mitigation = HallucinationMitigation()
        return self._hallucination_mitigation

    @property
    def legal_response_pipeline(self):
        """Lazy-loaded legal response pipeline."""
        if self._legal_response_pipeline is None:
            from src.models.hallucination_pipeline import LegalResponsePipeline
            logger.info("Initializing shared LegalResponsePipeline")
            self._legal_response_pipeline = LegalResponsePipeline(resource_manager=self)
        return self._legal_response_pipeline

    @property
    def basic_rag(self):
        """Lazy-loaded Basic RAG system."""
        if self._basic_rag is None:
            from src.retrieval.basic_rag import BasicRAG
            logger.info("Initializing shared BasicRAG")
            self._basic_rag = BasicRAG(resource_manager=self)
        return self._basic_rag

    @property
    def multi_query_rag(self):
        """Lazy-loaded Multi-Query RAG system."""
        if self._multi_query_rag is None:
            from src.retrieval.multi_query_rag import MultiQueryRAG
            logger.info("Initializing shared MultiQueryRAG")
            self._multi_query_rag = MultiQueryRAG(resource_manager=self)
        return self._multi_query_rag

    @property
    def metadata_enhanced_rag(self):
        """Lazy-loaded Metadata-Enhanced RAG system."""
        if self._metadata_enhanced_rag is None:
            from src.retrieval.metadata_enhanced_rag import MetadataEnhancedRAG
            logger.info("Initializing shared MetadataEnhancedRAG")
            self._metadata_enhanced_rag = MetadataEnhancedRAG(resource_manager=self)
        return self._metadata_enhanced_rag

    @property
    def advanced_rag(self):
        """Lazy-loaded Advanced RAG system."""
        if self._advanced_rag is None:
            from src.retrieval.advanced_rag import AdvancedRAG
            logger.info("Initializing shared AdvancedRAG")
            self._advanced_rag = AdvancedRAG(resource_manager=self)
        return self._advanced_rag