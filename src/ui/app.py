"""
LegalMind UI Application

This module implements a Streamlit UI for the LegalMind legal assistant
with support for all advanced features including RAG strategies,
hallucination mitigation, and RLHF.
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
from src.retrieval.query_expansion import LegalQueryExpansion
from src.retrieval.multi_query_rag import MultiQueryRAG
from src.retrieval.metadata_enhanced_rag import MetadataEnhancedRAG
from src.retrieval.advanced_rag import AdvancedRAG
from src.retrieval.metadata_filter import MetadataFilter
from src.models.llm_api import LMStudioAPI
from src.vectordb.chroma_db import ChromaVectorStore
from src.models.hallucination_pipeline import LegalResponsePipeline

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
        # Initialize RAG systems
        self.basic_rag = BasicRAG()
        self.query_expansion = LegalQueryExpansion()
        self.multi_query_rag = MultiQueryRAG()
        self.metadata_enhanced_rag = MetadataEnhancedRAG()
        self.advanced_rag = AdvancedRAG()

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

        # Initialize hallucination pipeline
        try:
            self.response_pipeline = LegalResponsePipeline()
            self.pipeline_available = True
            logger.info("Initialized hallucination pipeline")
        except Exception as e:
            logger.error(f"Error initializing hallucination pipeline: {str(e)}")
            self.response_pipeline = None
            self.pipeline_available = False

        # Set up session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "retrieved_docs" not in st.session_state:
            st.session_state.retrieved_docs = []

        if "filters" not in st.session_state:
            st.session_state.filters = {}

        if "response_analysis" not in st.session_state:
            st.session_state.response_analysis = None

        if "show_confidence" not in st.session_state:
            st.session_state.show_confidence = config["ui"]["features"]["confidence_display"]

        if "alternative_responses" not in st.session_state:
            st.session_state.alternative_responses = []

        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = False

        if "rag_strategy" not in st.session_state:
            st.session_state.rag_strategy = "advanced"

        if "rag_explanation" not in st.session_state:
            st.session_state.rag_explanation = None

        if "hallucination_mitigation_enabled" not in st.session_state:
            st.session_state.hallucination_mitigation_enabled = config["hallucination"]["enabled"]

        if "generate_alternatives" not in st.session_state:
            st.session_state.generate_alternatives = False

        if "show_rag_explanation" not in st.session_state:
            st.session_state.show_rag_explanation = config["ui"]["features"].get("rag_strategy_display", False)

        logger.info("Initialized LegalMind UI")

    def _format_document_preview(self, document: Dict[str, Any]) -> str:
        """Format a document for display in the UI."""
        preview = document["text"]

        # Truncate if too long
        if len(preview) > 300:
            preview = preview[:300] + "..."

        # Add metadata
        metadata_str = ""
        if "metadata" in document and document["metadata"]:
            for key, value in document["metadata"].items():
                if key != "embedding" and isinstance(value, (str, int, float, bool)):
                    metadata_str += f"**{key}**: {value}  \n"

        # Add relevance scores if available
        scores_str = ""
        if "similarity_score" in document:
            scores_str += f"**Similarity Score**: {document['similarity_score']:.2f}  \n"
        if "court_score" in document:
            scores_str += f"**Court Hierarchy Score**: {document['court_score']:.2f}  \n"
        if "recency_score" in document:
            scores_str += f"**Recency Score**: {document['recency_score']:.2f}  \n"
        if "combined_score" in document:
            scores_str += f"**Combined Score**: {document['combined_score']:.2f}  \n"
        if "perspective" in document:
            scores_str += f"**Query Perspective**: {document['perspective']}  \n"

        return f"{metadata_str}\n{scores_str}\n{preview}"

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

        # RAG Strategy Selection
        st.sidebar.subheader("Retrieval Strategy")

        rag_options = {
            "basic": "Basic RAG",
            "query_expansion": "Query Expansion RAG",
            "multi_query": "Multi-Query RAG",
            "metadata_enhanced": "Metadata-Enhanced RAG",
            "advanced": "Auto-Select (Advanced)"
        }

        selected_strategy = st.sidebar.selectbox(
            "Select Retrieval Strategy",
            list(rag_options.keys()),
            format_func=lambda x: rag_options[x],
            index=list(rag_options.keys()).index(st.session_state.rag_strategy),
            key="strategy_select"  # Added unique key
        )

        st.session_state.rag_strategy = selected_strategy

        # Display strategy description
        if selected_strategy == "basic":
            st.sidebar.info("Basic RAG retrieves documents using simple semantic similarity.")
        elif selected_strategy == "query_expansion":
            st.sidebar.info("Query Expansion enhances retrieval by adding legal synonyms and terminology.")
        elif selected_strategy == "multi_query":
            st.sidebar.info("Multi-Query captures different perspectives for the same legal question.")
        elif selected_strategy == "metadata_enhanced":
            st.sidebar.info("Metadata-Enhanced incorporates jurisdiction, court hierarchy, and recency.")
        elif selected_strategy == "advanced":
            st.sidebar.info("Advanced RAG automatically selects the best strategy for each query.")

        # Filter options
        st.sidebar.subheader("Metadata Filters")

        # Jurisdiction filter
        jurisdictions = self._get_jurisdictions()
        if jurisdictions:
            selected_jurisdiction = st.sidebar.selectbox(
                "Jurisdiction",
                ["All"] + jurisdictions,
                index=0,
                key="jurisdiction_filter"  # Added unique key
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
                index=0,
                key="doc_type_filter"  # Added unique key
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
            value=(1900, 2024),
            key="year_range_slider"  # Added unique key
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
            value=5,
            key="top_k_slider"  # Added unique key
        )

        # Temperature setting
        temperature = st.sidebar.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            key="temperature_slider"  # Added unique key
        )

        # Response Quality Settings
        st.sidebar.subheader("Response Quality")

        # Using unique keys for each checkbox
        st.session_state.show_confidence = st.sidebar.checkbox(
            "Show Confidence Scores",
            value=st.session_state.show_confidence,
            key="show_confidence_checkbox"
        )

        st.session_state.hallucination_mitigation_enabled = st.sidebar.checkbox(
            "Enable Hallucination Mitigation",
            value=st.session_state.hallucination_mitigation_enabled,
            key="hallucination_mitigation_checkbox"
        )

        st.session_state.generate_alternatives = st.sidebar.checkbox(
            "Generate Alternative Responses",
            value=st.session_state.generate_alternatives,
            key="generate_alternatives_checkbox"
        )

        st.session_state.show_rag_explanation = st.sidebar.checkbox(
            "Show RAG Strategy Explanation",
            value=st.session_state.show_rag_explanation,
            key="show_rag_explanation_checkbox"
        )

        # About section
        st.sidebar.markdown("---")
        st.sidebar.subheader("About LegalMind")
        st.sidebar.info(
            "LegalMind is an AI assistant designed to help users understand legal information "
            "and reasoning based on Australian law. It uses advanced retrieval-augmented generation "
            "with hallucination mitigation to provide accurate and contextual responses."
            "\n\n"
            "**Note**: LegalMind provides information, not legal advice. Always consult a "
            "qualified legal professional for advice specific to your situation."
        )

    def _display_messages(self):
        """Display the conversation history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Display confidence score if available and enabled
                if (message["role"] == "assistant" and
                        "confidence" in message and
                        st.session_state.show_confidence):

                    confidence = message["confidence"]

                    # Format confidence as percentage
                    confidence_pct = int(confidence * 100)

                    # Choose color based on confidence level
                    if confidence >= 0.8:
                        confidence_color = "green"
                    elif confidence >= 0.6:
                        confidence_color = "orange"
                    else:
                        confidence_color = "red"

                    st.markdown(f"<span style='color:{confidence_color};font-size:0.8em;'>Confidence: {confidence_pct}%</span>", unsafe_allow_html=True)

    def _display_retrieved_docs(self, documents: List[Dict[str, Any]]):
        """Display the retrieved documents."""
        if not documents:
            return

        with st.expander("View Retrieved Legal Documents", expanded=False):
            for i, doc in enumerate(documents):
                st.markdown(f"### Document {i+1}")
                st.markdown(self._format_document_preview(doc))
                st.markdown("---")

    def _display_response_analysis(self, analysis):
        """Display the response analysis if available."""
        if not analysis or not st.session_state.show_confidence:
            return

        with st.expander("Response Analysis", expanded=False):
            # Overall confidence score
            confidence_pct = int(analysis.get("confidence_score", 0) * 100)
            st.markdown(f"**Overall Confidence Score**: {confidence_pct}%")

            # Citation analysis
            citation_analysis = analysis.get("citation_analysis", {})
            if citation_analysis:
                st.markdown("### Citation Analysis")

                citation_results = citation_analysis.get("verification_results", [])
                if citation_results:
                    for i, result in enumerate(citation_results):
                        citation = result.get("citation", "")
                        verified = result.get("verified_in_context", False)
                        confidence = result.get("context_confidence", 0)

                        status = "✅ Verified" if verified else "❌ Not Verified"
                        st.markdown(f"**Citation {i+1}**: {citation}  \n{status} (Confidence: {int(confidence*100)}%)")

                verification_rate = citation_analysis.get("verification_rate", 1.0)
                st.markdown(f"**Citation Verification Rate**: {int(verification_rate*100)}%")

            # Hallucination analysis
            hallucination_analysis = analysis.get("hallucination_analysis", {})
            if hallucination_analysis:
                st.markdown("### Hallucination Analysis")

                has_hallucinations = hallucination_analysis.get("has_hallucinations", False)
                severity = hallucination_analysis.get("hallucination_severity", "none")

                status = "✅ No hallucinations detected" if not has_hallucinations else f"⚠️ Potential hallucinations detected (Severity: {severity})"
                st.markdown(status)

                # Jurisdiction analysis
                jurisdiction_analysis = hallucination_analysis.get("jurisdiction_analysis", {})
                if not jurisdiction_analysis.get("jurisdiction_match", True):
                    mismatched = jurisdiction_analysis.get("mismatched_jurisdictions", [])
                    st.markdown(f"⚠️ **Jurisdiction Mismatch**: Response discusses jurisdictions not supported by context: {', '.join(mismatched)}")

    def _display_rag_explanation(self, explanation):
        """Display the RAG strategy explanation if available."""
        if not explanation:
            return

        with st.expander("RAG Strategy Explanation", expanded=False):
            st.markdown(f"**Selected Strategy**: {explanation.get('best_strategy', 'basic')}")

            # Strategy-specific details
            if "query_expansion" in explanation:
                st.markdown("### Query Expansion")
                expanded_queries = explanation["query_expansion"].get("expanded_queries", [])

                if expanded_queries:
                    st.markdown("**Expanded Queries:**")
                    for i, query in enumerate(expanded_queries):
                        st.markdown(f"{i+1}. {query}")

                legal_terms = explanation["query_expansion"].get("legal_terms", [])
                if legal_terms:
                    st.markdown(f"**Identified Legal Terms**: {', '.join(legal_terms)}")

            elif "multi_query" in explanation:
                st.markdown("### Multi-Query Perspectives")
                perspectives = explanation["multi_query"].get("perspectives", [])

                if perspectives:
                    st.markdown("**Query Perspectives:**")
                    for i, perspective in enumerate(perspectives):
                        st.markdown(f"{i+1}. {perspective}")

            elif "metadata_enhanced" in explanation:
                st.markdown("### Metadata Enhancement")
                jurisdiction = explanation["metadata_enhanced"].get("jurisdiction", "")

                if jurisdiction:
                    st.markdown(f"**Identified Jurisdiction**: {jurisdiction}")

            # Result statistics
            if "results_by_strategy" in explanation:
                st.markdown("### Results by Strategy")

                for strategy, results in explanation["results_by_strategy"].items():
                    num_docs = results.get("num_docs", 0)
                    st.markdown(f"**{strategy}**: {num_docs} documents retrieved")

    def _handle_response_feedback(self, query, responses):
        """Handle user feedback on responses."""
        if not self.pipeline_available or not self.response_pipeline:
            return

        # Check if we have alternative responses to collect feedback on
        if len(responses) <= 1:
            return

        # If feedback already submitted, don't show again
        if st.session_state.feedback_submitted:
            return

        with st.form("feedback_form"):
            st.markdown("### Which response did you prefer?")

            # Radio buttons for selecting preferred response
            chosen_idx = st.radio(
                "Select the most helpful response:",
                options=list(range(1, len(responses) + 1)),
                format_func=lambda x: f"Response {x}",
                key="feedback_radio"  # Added unique key
            ) - 1  # Convert to 0-indexed

            # Text area for additional feedback
            feedback_text = st.text_area(
                "Optional feedback (what made this response better?):",
                height=100,
                key="feedback_text"  # Added unique key
            )

            # Submit button
            submitted = st.form_submit_button("Submit Feedback")

            if submitted:
                # Collect feedback using RLHF component
                self.response_pipeline.collect_feedback(
                    query,
                    responses,
                    chosen_idx,
                    feedback_text
                )

                st.success("Thank you for your feedback! It will help improve future responses.")
                st.session_state.feedback_submitted = True

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

            # Determine which RAG strategy to use
            rag_strategy = st.session_state.rag_strategy

            # Use retrieval with appropriate strategy
            thinking_placeholder.markdown(f"Retrieving relevant documents using {rag_strategy} strategy...")

            if rag_strategy == "basic":
                context, retrieved_docs = self.basic_rag.process_query(query)
                rag_explanation = None
            elif rag_strategy == "query_expansion":
                # Expand the query
                expanded_queries = self.query_expander.expand_query(query)
                expanded_query = expanded_queries[0] if expanded_queries else query

                # Use basic RAG with expanded query
                context, retrieved_docs = self.basic_rag.process_query(expanded_query)

                # Create explanation
                rag_explanation = {
                    "best_strategy": "query_expansion",
                    "query_expansion": {
                        "original_query": query,
                        "expanded_queries": expanded_queries,
                        "legal_terms": self.query_expander.identify_legal_terms(query)
                    }
                }
            elif rag_strategy == "multi_query":
                context, retrieved_docs = self.multi_query_rag.process_query(query)

                # Create explanation
                rag_explanation = {
                    "best_strategy": "multi_query",
                    "multi_query": {
                        "perspectives": self.multi_query_rag._generate_query_perspectives(query)
                    }
                }
            elif rag_strategy == "metadata_enhanced":
                context, retrieved_docs = self.metadata_enhanced_rag.process_query(query)

                # Create explanation
                rag_explanation = {
                    "best_strategy": "metadata_enhanced",
                    "metadata_enhanced": {
                        "jurisdiction": self.query_expander.identify_jurisdiction(query)
                    }
                }
            else:  # advanced (auto-select)
                context, retrieved_docs = self.advanced_rag.process_query(query)

                # Get strategy explanation
                rag_explanation = self.advanced_rag.explain_retrieval(query)

            # Apply metadata filters if any
            if st.session_state.filters:
                thinking_placeholder.markdown("Applying metadata filters...")

                # Use metadata filtering
                filtered_docs, filters = self.metadata_filter.filter_query(
                    query,
                    explicit_filters=st.session_state.filters
                )

                # If we have results after filtering, use them
                if filtered_docs:
                    context = self.basic_rag.prepare_context(filtered_docs)
                    retrieved_docs = filtered_docs

            # Store retrieved documents and RAG explanation
            st.session_state.retrieved_docs = retrieved_docs
            st.session_state.rag_explanation = rag_explanation

            # Generate response
            thinking_placeholder.markdown("Generating response...")

            if not self.llm:
                response = "Sorry, I cannot generate a response as the LM Studio API is not available. Please check the connection to LM Studio and try again."
                thinking_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            response = self.llm.generate(query, context)

            # Generate alternative responses if requested
            alternatives = []
            if st.session_state.generate_alternatives:  # Use session state variable instead of checkbox
                thinking_placeholder.markdown("Generating alternative responses...")

                # Generate two additional responses with different temperatures
                try:
                    alt_response1 = self.llm.generate(query, context)
                    # Use higher temperature for more diversity
                    temp_backup = self.llm.temperature
                    self.llm.temperature = 0.3
                    alt_response2 = self.llm.generate(query, context)
                    self.llm.temperature = temp_backup

                    alternatives = [response, alt_response1, alt_response2]
                    st.session_state.alternative_responses = alternatives
                except Exception as e:
                    logger.error(f"Error generating alternative responses: {str(e)}")
                    alternatives = []

            # Process through hallucination pipeline if available
            response_analysis = None
            improved_response = response
            confidence_score = 1.0

            if self.pipeline_available and self.response_pipeline and st.session_state.hallucination_mitigation_enabled:
                thinking_placeholder.markdown("Analyzing response for accuracy...")

                try:
                    pipeline_result = self.response_pipeline.generate_improved_response(
                        query, context, response
                    )

                    improved_response = pipeline_result.get("improved_response", response)
                    confidence_score = pipeline_result.get("confidence_score", 1.0)
                    response_analysis = pipeline_result

                    # Store analysis for display
                    st.session_state.response_analysis = response_analysis

                except Exception as e:
                    logger.error(f"Error in hallucination pipeline: {str(e)}")
                    # Fall back to original response
                    improved_response = response
                    confidence_score = 0.7

            # Update message with response
            thinking_placeholder.markdown(improved_response)

        # Add assistant message to chat with confidence score
        st.session_state.messages.append({
            "role": "assistant",
            "content": improved_response,
            "confidence": confidence_score
        })

        # Display retrieved documents
        self._display_retrieved_docs(retrieved_docs)

        # Display RAG explanation if enabled
        if st.session_state.show_rag_explanation and rag_explanation:
            self._display_rag_explanation(rag_explanation)

        # Display response analysis if available
        if response_analysis:
            self._display_response_analysis(response_analysis)

        # Handle response feedback if we have alternatives
        if alternatives:
            self._handle_response_feedback(query, alternatives)
        else:
            st.session_state.feedback_submitted = False

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