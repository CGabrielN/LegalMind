"""
LegalMind UI Application

This module implements a Streamlit UI for the LegalMind legal assistant.
"""

import os
import yaml
import logging
import sys
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path to import project modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Now use absolute imports
from src.retrieval.basic_rag import BasicRAG
from src.retrieval.metadata_filter import MetadataFilter
from src.models.llm_api import LMStudioAPI
from src.vectordb.chroma_db import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config_path = project_root / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set page title and icon
st.set_page_config(
    page_title=config["ui"]["title"],
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LegalMindUI:
    """
    Streamlit UI for the LegalMind legal assistant.
    """

    def __init__(self):
        """Initialize the UI components."""
        # Initialize RAG system
        self.rag = BasicRAG()

        # Initialize metadata filter
        self.metadata_filter = MetadataFilter()

        # Initialize LM Studio API connection
        try:
            # Get LM Studio API URL from config or use default
            lm_studio_url = config.get("lm_studio", {}).get("api_base_url", "http://127.0.0.1:1234/v1")
            self.llm = LMStudioAPI(api_base_url=lm_studio_url)
            self.llm_available = True
            st.success(f"Successfully connected to LM Studio API at {lm_studio_url}")
        except Exception as e:
            st.error(f"Error connecting to LM Studio API: {str(e)}")
            st.warning("Make sure LM Studio is running with a model loaded. Running in retrieval-only mode.")
            self.llm = None
            self.llm_available = False

        # Initialize vector store
        self.vector_store = ChromaVectorStore()

        # Set up session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "retrieved_docs" not in st.session_state:
            st.session_state.retrieved_docs = []

        if "filters" not in st.session_state:
            st.session_state.filters = {}

        logger.info("Initialized LegalMind UI")

    def _format_document_preview(self, document: Dict[str, Any]) -> str:
        """Format a document for display in the UI."""
        preview = document["text"]

        # Truncate if too long
        if len(preview) > 300:
            preview = preview[:300] + "..."

        # Add metadata
        metadata_str = ""
        if document["metadata"]:
            for key, value in document["metadata"].items():
                if key != "embedding" and isinstance(value, (str, int, float, bool)):
                    metadata_str += f"**{key}**: {value}  \n"

        return f"{metadata_str}\n{preview}"

    def _get_jurisdictions(self) -> List[str]:
        """Get available jurisdictions for filtering."""
        # Get values from metadata filter
        return sorted(list(self.metadata_filter.get_available_metadata_values("jurisdiction")))

    def _get_document_types(self) -> List[str]:
        """Get available document types for filtering."""
        # Get values from metadata filter
        return sorted(list(self.metadata_filter.get_available_metadata_values("document_type")))

    def _display_sidebar(self):
        """Display the sidebar with filters and options."""
        st.sidebar.title("LegalMind Options")

        # Display document count
        doc_count = self.vector_store.count_documents()
        st.sidebar.metric("Legal Documents in Database", doc_count)

        # Filter options
        st.sidebar.subheader("Filters")

        # Jurisdiction filter
        jurisdictions = self._get_jurisdictions()
        if jurisdictions:
            selected_jurisdiction = st.sidebar.selectbox(
                "Jurisdiction",
                ["All"] + jurisdictions,
                index=0
            )

            if selected_jurisdiction != "All":
                st.session_state.filters["jurisdiction"] = selected_jurisdiction
            elif "jurisdiction" in st.session_state.filters:
                del st.session_state.filters["jurisdiction"]

        # Document type filter
        doc_types = self._get_document_types()
        if doc_types:
            selected_doc_type = st.sidebar.selectbox(
                "Document Type",
                ["All"] + doc_types,
                index=0
            )

            if selected_doc_type != "All":
                st.session_state.filters["document_type"] = selected_doc_type
            elif "document_type" in st.session_state.filters:
                del st.session_state.filters["document_type"]

        # Year range filter
        st.sidebar.subheader("Year Range")
        years = list(range(1900, 2025))
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=1900,
            max_value=2024,
            value=(1900, 2024)
        )

        if year_range != (1900, 2024):
            st.session_state.filters["year"] = {
                "$gte": str(year_range[0]),
                "$lte": str(year_range[1])
            }
        elif "year" in st.session_state.filters:
            del st.session_state.filters["year"]

        # Advanced options
        st.sidebar.subheader("Advanced Options")

        # Number of documents to retrieve
        top_k = st.sidebar.slider(
            "Number of Documents to Retrieve",
            min_value=1,
            max_value=20,
            value=5
        )

        # Temperature setting
        temperature = st.sidebar.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1
        )

        # About section
        st.sidebar.markdown("---")
        st.sidebar.subheader("About LegalMind")
        st.sidebar.info(
            "LegalMind is an AI assistant designed to help users understand legal information "
            "and reasoning based on Australian law. It uses retrieval-augmented generation "
            "to provide accurate and contextual responses."
            "\n\n"
            "**Note**: LegalMind provides information, not legal advice. Always consult a "
            "qualified legal professional for advice specific to your situation."
        )

    def _display_messages(self):
        """Display the conversation history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _display_retrieved_docs(self, documents: List[Dict[str, Any]]):
        """Display the retrieved documents."""
        if not documents:
            return

        with st.expander("View Retrieved Legal Documents", expanded=False):
            for i, doc in enumerate(documents):
                st.markdown(f"### Document {i+1}")
                st.markdown(self._format_document_preview(doc))
                st.markdown("---")

    def process_query(self, query: str):
        """
        Process a legal query and generate a response.

        Args:
            query: The legal query
        """
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})

        # Display thinking indicator
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("Thinking...")

            # First check if we have explicit filters
            if st.session_state.filters:
                # Use metadata filtering
                retrieved_docs, filters = self.metadata_filter.filter_query(
                    query,
                    explicit_filters=st.session_state.filters
                )

                # If no results with filters, fall back to basic retrieval
                if not retrieved_docs:
                    thinking_placeholder.markdown("No results with filters, trying without filters...")
                    context, retrieved_docs = self.rag.process_query(query)
                else:
                    # Prepare context for LLM
                    context = self.rag.prepare_context(retrieved_docs)
            else:
                # Use basic RAG
                context, retrieved_docs = self.rag.process_query(query)

            # Store retrieved documents
            st.session_state.retrieved_docs = retrieved_docs

            # Generate response
            thinking_placeholder.markdown("Generating response...")
            response = self.llm.generate(query, context)

            # Update message with response
            thinking_placeholder.markdown(response)

        # Add assistant message to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display retrieved documents
        self._display_retrieved_docs(retrieved_docs)

    def run(self):
        """Run the Streamlit application."""
        # Display title
        st.title("🏛️ LegalMind Assistant")
        st.subheader("Your AI guide to Australian legal information")

        # Display sidebar
        self._display_sidebar()

        # Display chat messages
        self._display_messages()

        # Display chat input
        if query := st.chat_input("Ask a legal question about Australian law..."):
            self.process_query(query)

        # Welcome message for new conversations
        if not st.session_state.messages:
            st.info(
                "👋 Welcome to LegalMind! I'm here to help you understand Australian legal "
                "concepts and information. Ask me questions about Australian law, legal cases, "
                "or legal principles, and I'll provide information based on legal precedents "
                "and statutes."
                "\n\n"
                "For example, you can ask:"
                "\n\n"
                "- What constitutes negligence in New South Wales?"
                "\n"
                "- Explain the legal principles in contract law regarding offer and acceptance."
                "\n"
                "- What are my rights as a tenant in Victoria?"
                "\n\n"
                "**Note**: I provide legal information, not legal advice. For advice specific to "
                "your situation, please consult a qualified legal professional."
            )


if __name__ == "__main__":
    # Initialize and run the UI
    app = LegalMindUI()
    app.run()