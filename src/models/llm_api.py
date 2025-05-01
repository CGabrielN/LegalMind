"""
LegalMind LM Studio API Integration

This module handles the interaction with LM Studio's local API
for text generation using retrieved context.
"""

import os
import yaml
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Generator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(project_root, "config", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

class LMStudioAPI:
    """
    Handles interaction with LM Studio's local API for legal question answering.
    """

    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize the API connection."""
        self.api_base_url = api_base_url or "http://127.0.0.1:1234/v1"
        self.completions_url = f"{self.api_base_url}/completions"
        self.chat_completions_url = f"{self.api_base_url}/chat/completions"
        self.model_name = "phi-4-mini-instruct"  # This is arbitrary as LM Studio uses the loaded model
        self.temperature = config["llm"].get("temperature", 0.1)
        self.max_tokens = config["llm"].get("max_new_tokens", 1024)

        logger.info(f"Initialized LM Studio API connection to {self.api_base_url}")

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test the connection to the LM Studio API."""
        try:
            response = requests.get(self.api_base_url)
            if response.status_code == 404:
                # This is actually expected - the root endpoint doesn't exist
                # but tells us the server is running
                logger.info("LM Studio API is reachable")
            else:
                logger.info(f"LM Studio API responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to LM Studio API at {self.api_base_url}")
            logger.error("Please ensure LM Studio is running and the API is enabled.")
            raise ConnectionError(f"Failed to connect to LM Studio API at {self.api_base_url}")

    def _create_prompt(self, query: str, context: str) -> Dict[str, Any]:
        """
        Create a prompt for the LLM based on the query and retrieved context.

        Args:
            query: The user's legal query
            context: Retrieved legal context

        Returns:
            Formatted chat messages
        """
        system_message = (
            "You are LegalMind, an AI legal assistant specializing in Australian law. "
            "You provide accurate information based on legal precedents and statutes. "
            "When answering questions, cite relevant cases and explain legal concepts clearly. "
            "You should explain complex legal terms in plain language. "
            "If you don't know something or if the information isn't in the provided context, "
            "acknowledge that and don't make up information."
        )

        user_message = f"Context information is below:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        return messages

    def generate(self, query: str, context: str) -> str:
        """
        Generate a response using the LLM via LM Studio API.

        Args:
            query: The user's legal query
            context: Retrieved legal context

        Returns:
            Generated response
        """
        logger.info(f"Generating response for query: '{query[:50]}...'")

        # Create the prompt as chat messages
        messages = self._create_prompt(query, context)

        # Prepare the API request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Make the API call
            response = requests.post(
                self.chat_completions_url,
                headers=headers,
                data=json.dumps(payload)
            )

            # Check for errors
            response.raise_for_status()

            # Parse the response
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                if "content" in message:
                    return message["content"].strip()

            # If we couldn't get the response content, return an error
            logger.error(f"Unexpected API response structure: {result}")
            return "I apologize, but I encountered an issue while generating a response to your legal question. Please try again."

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LM Studio API: {str(e)}")
            return "I apologize, but I encountered an error while connecting to the language model. Please ensure LM Studio is running correctly."

    def generate_stream(self, query: str, context: str) -> Generator[str, None, None]:
        """
        Generate a streaming response using the LLM via LM Studio API.

        Args:
            query: The user's legal query
            context: Retrieved legal context

        Returns:
            Generator yielding response tokens
        """
        logger.info(f"Generating streaming response for query: '{query[:50]}...'")

        # Create the prompt as chat messages
        messages = self._create_prompt(query, context)

        # Prepare the API request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Make the API call with streaming enabled
            response = requests.post(
                self.chat_completions_url,
                headers=headers,
                data=json.dumps(payload),
                stream=True
            )

            # Check for errors
            response.raise_for_status()

            # Process streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    yield content
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON: {data}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error streaming from LM Studio API: {str(e)}")
            yield "I apologize, but I encountered an error while connecting to the language model."