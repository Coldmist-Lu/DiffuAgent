"""
API_LLM - Generic OpenAI-Format API Client.

A flexible LLM backend that connects to OpenAI-format APIs with:
- Multi-port automatic detection
- Token counting support
- Throughput monitoring
- Standard OpenAI API format
"""
import os
import time
import requests
from common.registry import registry
from utils.logging.agent_logger import AgentLogger
import logging

# Module logger
logger = AgentLogger("API_LLM")
logger.setLevel(logging.INFO)


# Port configuration for API servers
# Automatically detects the first available server
PORTS_CONFIG = {
    "0": {
        "API_KEY": os.getenv("VLLM_API_KEY", "your-api-key-here"),
        "BASE_URL": os.getenv("VLLM_BASE_URL", "http://localhost:23456/"),
    },
    "1": {
        "API_KEY": os.getenv("VLLM_API_KEY", "your-api-key-here"),
        "BASE_URL": os.getenv("VLLM_BASE_URL_1", "http://localhost:23460/"),
    },
    "2": {
        "API_KEY": os.getenv("VLLM_API_KEY", "your-api-key-here"),
        "BASE_URL": os.getenv("VLLM_BASE_URL_2", "http://localhost:23465/"),
    },
}


@registry.register_llm("api_llm")
class API_LLM:
    """
    Generic OpenAI-format API client with multi-port detection.

    Features:
    - Automatically detects available server from configured ports
    - Supports standard OpenAI API format (/v1/chat/completions)
    - Optional token counting
    - Throughput monitoring

    Configuration:
        llm_config:
            engine: Model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum tokens to generate (default: 256)
            context_length: Model context length (default: 4096)
            return_token: Return token count with response (default: False)
            api_key: Override API key (optional)
            base_url: Override base URL (optional)

    Example:
        llm_config = {
            "engine": "meta-llama/Llama-2-7b-chat-hf",
            "temperature": 0.1,
            "max_tokens": 256,
            "return_token": True
        }
        llm = API_LLM.from_config(llm_config)
    """

    def __init__(self,
                 engine="",
                 temperature=0.1,
                 max_tokens=256,
                 context_length=4096,
                 return_token=False,
                 api_key=None,
                 base_url=None):

        self.engine = engine
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.return_token = return_token

        # Detect available server if not explicitly provided
        if base_url is None or api_key is None:
            for port_id, port_params in PORTS_CONFIG.items():
                if self._check_model_availability(
                    port_params["BASE_URL"],
                    port_params["API_KEY"],
                    engine
                ):
                    self.base_url = port_params["BASE_URL"]
                    self.api_key = port_params["API_KEY"]
                    logger.info(f"Server found at {self.base_url}")
                    break
            else:
                raise Exception("API_LLM: No available server detected")
        else:
            self.base_url = base_url
            self.api_key = api_key
            logger.info(f"Using configured server at {self.base_url}")

        self.chat_url = self.base_url + 'v1/chat/completions'
        self.tokenize_url = self.base_url + 'tokenize'

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.request_template = {
            "model": engine,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        self.tokenizer_template = {
            "model": engine,
        }

    def llm_inference(self, messages):
        """
        Perform LLM inference with messages.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            tuple: (response_text, token_count)
        """
        data = self.request_template.copy()
        data['messages'] = messages

        start_time = time.time()
        response = requests.post(url=self.chat_url, headers=self.headers, json=data)
        end_time = time.time()

        if response.status_code != 200:
            raise Exception(f"API_LLM inference failed: {response.status_code} - {response.text}")

        response_dict = response.json()
        throughput = response_dict['usage']['completion_tokens'] / (end_time - start_time)
        logger.debug(f"Throughput: {throughput:.2f} tokens/s")

        return (
            response_dict['choices'][0]['message']['content'],
            int(response_dict['usage']['completion_tokens'])
        )

    def generate(self, prompt):
        """
        Generate completion from prompt.

        Args:
            prompt: List of messages in OpenAI format

        Returns:
            tuple: (success, response) or (success, (response, token_count))
        """
        response, token = self.llm_inference(prompt)

        if self.return_token:
            return True, (response, token)
        else:
            return True, response

    def num_tokens_from_messages(self, messages):
        """
        Count tokens in messages using the tokenizer endpoint.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            int: Number of tokens
        """
        data = self.tokenizer_template.copy()
        data['messages'] = messages

        response = requests.post(self.tokenize_url, headers=self.headers, json=data)

        if response.status_code != 200:
            raise Exception(f"API_LLM tokenization failed: {response.status_code} - {response.text}")

        response_dict = response.json()
        return response_dict['count']

    def _check_model_availability(self, base_url, api_key, engine):
        """
        Check if the model is available at the given URL.

        Args:
            base_url: Base URL of the server
            api_key: API key for authentication
            engine: Model name to check

        Returns:
            bool: True if available, False otherwise
        """
        test_url = base_url + 'tokenize'
        test_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        test_data = {
            "model": engine,
            "messages": [{"role": "user", "content": "Hello!"}]
        }

        try:
            response = requests.post(test_url, headers=test_headers, json=test_data, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def from_config(cls, config):
        """
        Create API_LLM instance from configuration dict.

        Args:
            config: Configuration dict with keys:
                - engine (required)
                - temperature (optional)
                - max_tokens (optional)
                - context_length (optional)
                - return_token (optional)
                - api_key (optional)
                - base_url (optional)

        Returns:
            API_LLM instance
        """
        engine = config.get("engine", "")
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_tokens", 256)
        context_length = config.get("context_length", 4096)
        return_token = config.get("return_token", False)
        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)

        return cls(
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            context_length=context_length,
            return_token=return_token,
            api_key=api_key,
            base_url=base_url
        )
