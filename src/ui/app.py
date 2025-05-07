"""
LegalMind UI Application

This module implements a Streamlit UI for the LegalMind legal assistant
with support for advanced features including RAG strategies and
hallucination mitigation.
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

# Add custom CSS for better UI elements
st.markdown("""
<style>
    /* Enhance button visibility */
    .feedback-button {
        margin: 5px;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: #f0f0f0;
        display: inline-block;
    }
    .feedback-button:hover {
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

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

        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = {}  # Track feedback by message index

        if "rag_strategy" not in st.session_state:
            st.session_state.rag_strategy = "advanced"

        if "rag_explanation" not in st.session_state:
            st.session_state.rag_explanation = None

        if "hallucination_mitigation_enabled" not in st.session_state:
            st.session_state.hallucination_mitigation_enabled = config["hallucination"]["enabled"]

        if "show_rag_explanation" not in st.session_state:
            st.session_state.show_rag_explanation = config["ui"]["features"].get("rag_strategy_display", False)

        if "top_k_value" not in st.session_state:
            st.session_state.top_k_value = 5

        if "temperature_value" not in st.session_state:
            st.session_state.temperature_value = 0.1

        # RLHF training status tracking
        if "last_rlhf_check" not in st.session_state:
            st.session_state.last_rlhf_check = None

        if "rlhf_training_pending" not in st.session_state:
            st.session_state.rlhf_training_pending = False

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
            key="strategy_select"
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
                key="jurisdiction_filter"
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
                key="doc_type_filter"
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
            key="year_range_slider"
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
            value=st.session_state.top_k_value,
            key="top_k_slider"
        )
        # Store immediately in session state for easier access
        st.session_state.top_k_value = top_k

        # Temperature setting
        temperature = st.sidebar.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature_value,
            step=0.1,
            key="temperature_slider"
        )

        # Store immediately in session state for easier access
        st.session_state.temperature_value = temperature

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

        st.session_state.show_rag_explanation = st.sidebar.checkbox(
            "Show RAG Strategy Explanation",
            value=st.session_state.show_rag_explanation,
            key="show_rag_explanation_checkbox"
        )

        # RLHF Training section
        st.sidebar.subheader("RLHF Training")

        # Display training status
        if self.pipeline_available and hasattr(self.response_pipeline, "rlhf"):
            try:
                rlhf_status = self.response_pipeline.rlhf.get_status()

                # Display collected feedback count
                feedback_count = rlhf_status["dataset"]["total_pairs"]
                min_required = rlhf_status["min_preference_pairs"]

                st.sidebar.metric("Collected Feedback", f"{feedback_count}/{min_required} pairs")

                # Show training status
                training_status = "Pending" if rlhf_status["pending_training"] else "Up to date"
                st.sidebar.text(f"Training Status: {training_status}")

                # Add train button
                if st.sidebar.button("Train RLHF Model"):
                    if feedback_count >= min_required:
                        with st.sidebar:
                            with st.spinner("Training RLHF model..."):
                                success = self.response_pipeline.rlhf.run_training_if_needed()
                                if success:
                                    st.success("RLHF training completed!")
                                else:
                                    st.warning("Training not needed or already in progress.")
                    else:
                        st.sidebar.warning(f"Need at least {min_required} feedback pairs to train.")

            except Exception as e:
                logger.error(f"Error getting RLHF status: {str(e)}")
                st.sidebar.warning("RLHF status unavailable")
        else:
            st.sidebar.warning("RLHF system not available")

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

    def _display_response(self, msg_idx, message):
        """
        Display a response with feedback buttons.

        Args:
            msg_idx: Message index in the conversation
            message: The message dictionary
        """
        # Generate a unique key for this response
        response_key = f"msg_{msg_idx}"

        # Get the response content
        response_content = message["content"]

        # Display response content
        st.markdown(response_content)

        # Display confidence score if available and enabled
        if "confidence" in message and st.session_state.show_confidence:
            confidence = message["confidence"]
            confidence_pct = int(confidence * 100)

            # Choose color based on confidence level
            if confidence >= 0.8:
                confidence_color = "green"
            elif confidence >= 0.6:
                confidence_color = "orange"
            else:
                confidence_color = "red"

            st.markdown(f"<span style='color:{confidence_color};font-size:0.8em;'>Confidence: {confidence_pct}%</span>", unsafe_allow_html=True)

        # Show feedback section
        feedback_container = st.container()
        with feedback_container:
            # Check if feedback was already submitted
            if response_key in st.session_state.feedback_submitted:
                # Show feedback status
                feedback = st.session_state.feedback_submitted[response_key]
                if feedback == "liked":
                    st.markdown("✅ You liked this response")
                elif feedback == "disliked":
                    st.markdown("❌ You disliked this response")
            else:
                # Create feedback buttons
                cols = st.columns([1, 1, 10])

                # Function to handle like button click
                def on_like_click():
                    if self.pipeline_available and self.response_pipeline:
                        self._handle_feedback(
                            message.get("query", ""),
                            response_content,
                            True
                        )
                        st.session_state.feedback_submitted[response_key] = "liked"

                        # Check if we should trigger RLHF training
                        if self.pipeline_available and hasattr(self.response_pipeline, "rlhf"):
                            try:
                                if self.response_pipeline.rlhf.pending_training:
                                    st.session_state.rlhf_training_pending = True
                            except:
                                pass

                # Function to handle dislike button click
                def on_dislike_click():
                    if self.pipeline_available and self.response_pipeline:
                        self._handle_feedback(
                            message.get("query", ""),
                            response_content,
                            False
                        )
                        st.session_state.feedback_submitted[response_key] = "disliked"

                        # Check if we should trigger RLHF training
                        if self.pipeline_available and hasattr(self.response_pipeline, "rlhf"):
                            try:
                                if self.response_pipeline.rlhf.pending_training:
                                    st.session_state.rlhf_training_pending = True
                            except:
                                pass

                # Add the buttons with callbacks
                with cols[0]:
                    if st.button("👍", key=f"like_{response_key}", on_click=on_like_click):
                        pass  # The actual action happens in the on_click function

                with cols[1]:
                    if st.button("👎", key=f"dislike_{response_key}", on_click=on_dislike_click):
                        pass  # The actual action happens in the on_click function

    def _display_messages(self):
        """Display the conversation history with feedback buttons."""
        for i, message in enumerate(st.session_state.messages):
            # Display standard messages
            if message["role"] in ["user", "system"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle assistant messages
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    # Display response content
                    st.markdown(message["content"])

                    # Generate a unique key for this response
                    feedback_key = f"msg_{i}"

                    # Display confidence score if available and enabled
                    if "confidence" in message and st.session_state.show_confidence:
                        confidence = message["confidence"]
                        confidence_pct = int(confidence * 100)

                        # Choose color based on confidence level
                        if confidence >= 0.8:
                            confidence_color = "green"
                        elif confidence >= 0.6:
                            confidence_color = "orange"
                        else:
                            confidence_color = "red"

                        st.markdown(f"<span style='color:{confidence_color};font-size:0.8em;'>Confidence: {confidence_pct}%</span>", unsafe_allow_html=True)

                    # Add feedback buttons directly here - always show for each message
                    st.write("") # Add a small space

                    # Check if feedback already submitted for this message
                    if feedback_key in st.session_state.feedback_submitted:
                        feedback = st.session_state.feedback_submitted[feedback_key]
                        if feedback == "liked":
                            st.success("✅ You liked this response")
                        elif feedback == "disliked":
                            st.error("❌ You disliked this response")
                    else:
                        # Show feedback buttons if no feedback yet submitted
                        feedback_cols = st.columns([1, 1, 10])

                        with feedback_cols[0]:
                            if st.button("👍", key=f"like_{i}"):
                                if self.pipeline_available and self.response_pipeline:
                                    self._handle_feedback(
                                        message.get("query", ""),
                                        message["content"],
                                        True
                                    )
                                    # Store feedback and force rerun
                                    st.session_state.feedback_submitted[feedback_key] = "liked"
                                    st.experimental_rerun()

                        with feedback_cols[1]:
                            if st.button("👎", key=f"dislike_{i}"):
                                if self.pipeline_available and self.response_pipeline:
                                    self._handle_feedback(
                                        message.get("query", ""),
                                        message["content"],
                                        False
                                    )
                                    # Store feedback and force rerun
                                    st.session_state.feedback_submitted[feedback_key] = "disliked"
                                    st.experimental_rerun()

    def _handle_feedback(self, query, response, is_positive):
        """
        Handle feedback for a response with like/dislike approach.

        Args:
            query: The original query
            response: The response being rated
            is_positive: True for like, False for dislike
        """
        if not self.pipeline_available or not self.response_pipeline:
            return

        # Use enhanced RLHF if available
        if hasattr(self.response_pipeline, "rlhf") and hasattr(self.response_pipeline.rlhf, "collect_like_dislike_feedback"):
            try:
                # Use the enhanced feedback collection
                self.response_pipeline.rlhf.collect_like_dislike_feedback(
                    query=query,
                    response=response,
                    is_positive=is_positive
                )
                logger.info(f"Collected {'positive' if is_positive else 'negative'} feedback for response")
                return
            except Exception as e:
                logger.error(f"Error collecting enhanced feedback: {str(e)}")
                # Fall back to basic method

        # Fallback to basic feedback collection
        feedback_text = "Positive user feedback" if is_positive else "Negative user feedback"

        if is_positive:
            # For positive feedback, the current response is chosen
            rejected = "This is a basic response without detail or citations."
            self.response_pipeline.collect_feedback(
                query,
                [response, rejected],
                chosen_idx=0,
                feedback_text=feedback_text
            )
        else:
            # For negative feedback, a better option is chosen
            chosen = "I apologize, but I need more specific information to provide an accurate response about Australian law. Could you please provide more details about your legal question?"
            self.response_pipeline.collect_feedback(
                query,
                [chosen, response],
                chosen_idx=0,
                feedback_text=feedback_text
            )

        logger.info(f"Collected {'positive' if is_positive else 'negative'} feedback using basic method")

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

            # Get values from session state
            top_k_value = st.session_state.top_k_value

            # Determine which RAG strategy to use
            rag_strategy = st.session_state.rag_strategy

            # Update the top_k parameter in all RAG strategies
            self.basic_rag.top_k = top_k_value
            if hasattr(self, 'query_expansion'):
                self.query_expansion.top_k = top_k_value
            if hasattr(self, 'multi_query_rag'):
                self.multi_query_rag.top_k = top_k_value
            if hasattr(self, 'metadata_enhanced_rag'):
                self.metadata_enhanced_rag.top_k = top_k_value
            if hasattr(self, 'advanced_rag'):
                self.advanced_rag.top_k = top_k_value

            # Use retrieval with appropriate strategy
            thinking_placeholder.markdown(f"Retrieving relevant documents using {rag_strategy} strategy...")

            if rag_strategy == "basic":
                context, retrieved_docs = self.basic_rag.process_query(query)
                rag_explanation = None
            elif rag_strategy == "query_expansion":
                # Expand the query
                expanded_queries = self.query_expansion.expand_query(query)
                expanded_query = expanded_queries[0] if expanded_queries else query

                # Use basic RAG with expanded query
                context, retrieved_docs = self.basic_rag.process_query(expanded_query)

                # Create explanation
                rag_explanation = {
                    "best_strategy": "query_expansion",
                    "query_expansion": {
                        "original_query": query,
                        "expanded_queries": expanded_queries,
                        "legal_terms": self.query_expansion.identify_legal_terms(query)
                    }
                }
            elif rag_strategy == "multi_query":
                context, retrieved_docs = self.multi_query_rag.process_query(query)

                # Create explanation
                rag_explanation = {
                    "best_strategy": "multi_query",
                    "multi_query": {
                        "perspectives": self.multi_query_rag.generate_query_perspectives(query)
                    }
                }
            elif rag_strategy == "metadata_enhanced":
                context, retrieved_docs = self.metadata_enhanced_rag.process_query(query)

                # Create explanation
                rag_explanation = {
                    "best_strategy": "metadata_enhanced",
                    "metadata_enhanced": {
                        "jurisdiction": self.query_expansion.identify_jurisdiction(query)
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

            # Update LLM temperature from session state
            self.llm.temperature = st.session_state.get("temperature_value", 0.1)
            response = self.llm.generate(query, context)

            # Process through hallucination pipeline if available
            response_analysis = None
            improved_response = response
            confidence_score = 1.0

            if self.pipeline_available and self.response_pipeline and st.session_state.hallucination_mitigation_enabled:
                thinking_placeholder.markdown("Analyzing response for accuracy...")

                try:
                    # Try to use enhanced RLHF to improve response if available
                    if hasattr(self.response_pipeline, "rlhf") and hasattr(self.response_pipeline.rlhf, "create_better_response"):
                        try:
                            # Use RLHF to potentially improve the response
                            thinking_placeholder.markdown("Using RLHF feedback to enhance response...")
                            rlhf_response = self.response_pipeline.rlhf.create_better_response(
                                query, response, context, self.llm
                            )
                            if rlhf_response != response:
                                logger.info("Response improved with RLHF")
                                response = rlhf_response
                        except Exception as e:
                            logger.error(f"Error using RLHF to improve response: {str(e)}")

                    # Apply hallucination mitigation
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

            # Add feedback buttons directly in this message context
            col1, col2, col3 = st.columns([1, 1, 10])

            # Generate a unique message ID for this response
            message_id = len(st.session_state.messages)
            feedback_key = f"msg_{message_id}"

            with col1:
                if st.button("👍", key=f"like_{feedback_key}"):
                    if self.pipeline_available and self.response_pipeline:
                        self._handle_feedback(query, improved_response, True)
                        st.session_state.feedback_submitted[feedback_key] = "liked"
                        st.success("Thanks for your positive feedback!")

            with col2:
                if st.button("👎", key=f"dislike_{feedback_key}"):
                    if self.pipeline_available and self.response_pipeline:
                        self._handle_feedback(query, improved_response, False)
                        st.session_state.feedback_submitted[feedback_key] = "disliked"
                        st.error("Thanks for your feedback. We'll improve!")

        # Add assistant message to chat with confidence score
        message = {
            "role": "assistant",
            "content": improved_response,
            "confidence": confidence_score,
            "query": query  # Store the query for feedback purposes
        }

        st.session_state.messages.append(message)

        # Display retrieved documents
        self._display_retrieved_docs(retrieved_docs)

        # Display RAG explanation if enabled
        if st.session_state.show_rag_explanation and rag_explanation:
            self._display_rag_explanation(rag_explanation)

        # Display response analysis if available
        if response_analysis:
            self._display_response_analysis(response_analysis)

        # Check if RLHF training is pending and show a notification
        if st.session_state.rlhf_training_pending and self.pipeline_available and hasattr(self.response_pipeline, "rlhf"):
            try:
                rlhf_status = self.response_pipeline.rlhf.get_status()
                if rlhf_status["pending_training"]:
                    st.info("📊 Enough feedback has been collected to train the RLHF model. Use the 'Train RLHF Model' button in the sidebar to improve response quality.")
            except:
                pass

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
            # Add user message directly to the UI first
            with st.chat_message("user"):
                st.markdown(query)
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
                "**Features**:"
                "\n"
                "- 👍/👎 Feedback: Rate responses to help improve future answers"
                "\n"
                "- 🧠 RLHF Learning: The system learns from your feedback to generate better responses"
                "\n\n"
                "**Note**: I provide legal information, not legal advice. For advice specific to "
                "your situation, please consult a qualified legal professional."
            )


if __name__ == "__main__":
    # Initialize and run the UI
    app = LegalMindUI()
    app.run()