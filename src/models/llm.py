"""
LegalMind LLM Module

This module handles the interaction with language models for text generation
using retrieved context.
"""

import os
import yaml
import logging
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class LegalLLM:
    """
    Handles interaction with LLMs for legal question answering.
    """

    def __init__(self):
        """Initialize the LLM with configuration."""
        self.model_name = config["llm"]["model_name"]
        self.device = config["llm"]["device"]
        self.temperature = config["llm"]["temperature"]
        self.max_new_tokens = config["llm"]["max_new_tokens"]
        self.use_flash_attention = config["llm"]["use_flash_attention"]

        # Check if the device is available
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.use_flash_attention = False

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the language model and tokenizer."""
        logger.info(f"Loading LLM: {self.model_name}")

        try:
            # Configure quantization for efficiency
            if self.device == "cuda":
                # Use BitsAndBytes for 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

                # Load the model with quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    use_flash_attention_2=self.use_flash_attention
                )
            else:
                # For CPU, load with minimal settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Successfully loaded model {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM based on the query and retrieved context.

        Args:
            query: The user's legal query
            context: Retrieved legal context

        Returns:
            Formatted prompt string
        """
        # Determine which model we're using to format the prompt correctly
        if "mistral" in self.model_name.lower():
            # Mistral uses this format
            system_prompt = (
                "You are LegalMind, an AI legal assistant specializing in Australian law. "
                "You provide accurate information based on legal precedents and statutes. "
                "When answering questions, cite relevant cases and explain legal concepts clearly. "
                "You should explain complex legal terms in plain language. "
                "If you don't know something or if the information isn't in the provided context, "
                "acknowledge that and don't make up information."
            )

            prompt = f"<s>[INST] {system_prompt}\n\nContext information is below:\n{context}\n\nQuestion: {query} [/INST]"

        elif "phi" in self.model_name.lower():
            # Phi uses this format
            system_prompt = (
                "You are LegalMind, an AI legal assistant specializing in Australian law. "
                "You provide accurate information based on legal precedents and statutes. "
                "When answering questions, cite relevant cases and explain legal concepts clearly. "
                "You should explain complex legal terms in plain language. "
                "If you don't know something or if the information isn't in the provided context, "
                "acknowledge that and don't make up information."
            )

            prompt = f"<|system|>\n{system_prompt}\n<|user|>\nContext information is below:\n{context}\n\nQuestion: {query}\n<|assistant|>\n"

        else:
            # Generic format for other models
            prompt = f"""
System: You are LegalMind, an AI legal assistant specializing in Australian law. 
You provide accurate information based on legal precedents and statutes.
When answering questions, cite relevant cases and explain legal concepts clearly.
You should explain complex legal terms in plain language.
If you don't know something or if the information isn't in the provided context, 
acknowledge that and don't make up information.

Context information is below:
{context}

Question: {query}

Answer:
"""

        return prompt

    def generate(self, query: str, context: str) -> str:
        """
        Generate a response using the LLM.

        Args:
            query: The user's legal query
            context: Retrieved legal context

        Returns:
            Generated response
        """
        logger.info(f"Generating response for query: '{query[:50]}...'")

        # Create the prompt
        prompt = self._create_prompt(query, context)

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the response
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0.0,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            logger.info(f"Generated response of length {len(response)}")

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response to your legal question. Please try again or rephrase your question."

    def generate_stream(self, query: str, context: str, callback=None):
        """
        Generate a streaming response using the LLM.

        Args:
            query: The user's legal query
            context: Retrieved legal context
            callback: Function to call with each token

        Returns:
            Generator yielding response tokens
        """
        logger.info(f"Generating streaming response for query: '{query[:50]}...'")

        # Create the prompt
        prompt = self._create_prompt(query, context)

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Create a streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate the response in a separate thread
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask", None),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0.0,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they're generated
        for text in streamer:
            if callback:
                callback(text)
            yield text

    def analyze_legal_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract and analyze legal citations from text.

        Args:
            text: Text containing legal citations

        Returns:
            List of dictionaries with citation information
        """
        import re

        # Pattern for Australian citations
        pattern = r"([A-Za-z\s]+\sv\s[A-Za-z\s]+\s\[\d{4}\]\s[A-Z]+\s\d+)"

        # Find all citations
        citations = re.findall(pattern, text)

        results = []
        for citation in citations:
            # Extract case name
            case_name_match = re.match(r"([A-Za-z\s]+\sv\s[A-Za-z\s]+)", citation)
            case_name = case_name_match.group(1) if case_name_match else ""

            # Extract year
            year_match = re.search(r"\[(\d{4})\]", citation)
            year = year_match.group(1) if year_match else ""

            # Extract court and number
            court_match = re.search(r"\[\d{4}\]\s([A-Z]+)\s(\d+)", citation)
            court = court_match.group(1) if court_match else ""
            number = court_match.group(2) if court_match else ""

            results.append({
                "citation": citation,
                "case_name": case_name,
                "year": year,
                "court": court,
                "number": number
            })

        return results