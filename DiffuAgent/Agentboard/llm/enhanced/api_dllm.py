"""
API_DiffusionLLM - DiffusionLLM API Client.

A specialized LLM backend for DiffusionLLM with:
- DiffusionLLM-specific parameters (steps, dual_cache, block_size, threshold)
- Token counting support
- Single-port configuration
"""
import os
import time
import requests
from common.registry import registry
from utils.logging.agent_logger import AgentLogger
import logging

# Module logger
logger = AgentLogger("API_DiffusionLLM")
logger.setLevel(logging.INFO)


# DiffusionLLM server configuration
# Set via environment variables: DLLM_API_KEY, DLLM_BASE_URL or FEATURES_API_KEY, FEATURES_BASE_URL
DLLM_CONFIG = {
    "API_KEY": os.getenv("DLLM_API_KEY") or os.getenv("FEATURES_API_KEY") or "",
    "BASE_URL": os.getenv("DLLM_BASE_URL") or os.getenv("FEATURES_BASE_URL") or "",
}


@registry.register_llm("api_dllm")
class API_DiffusionLLM:
    """
    DiffusionLLM API client with DiffusionLLM-specific parameters.

    Features:
    - DiffusionLLM-specific generation parameters
    - Token counting support
    - Optimized for DiffusionLLM models (Llada, Dream)

    Configuration:
        llm_config:
            engine: Model name (default: "Llada")
            gen_length: Maximum tokens to generate (default: 256)
            steps: Diffusion steps (default: 256)
            temperature: Sampling temperature (default: 0.0)
            dual_cache: Enable dual cache (default: False)
            block_size: Cache block size (default: 32)
            threshold: Cache threshold (default: 0.9)
            context_length: Model context length (default: 4000)
            return_token: Return token count with response (default: False)

    Example:
        llm_config = {
            "engine": "Llada",
            "gen_length": 256,
            "steps": 256,
            "temperature": 0.0,
            "dual_cache": False,
            "return_token": True
        }
        llm = API_DiffusionLLM.from_config(llm_config)
    """

    def __init__(self,
                 engine="Llada",
                 gen_length=256,
                 steps=256,
                 temperature=0.0,
                 dual_cache=False,
                 block_size=32,
                 threshold=0.9,
                 context_length=4000,
                 return_token=False):

        self.engine = engine
        self.context_length = context_length
        self.max_tokens = gen_length
        self.return_token = return_token

        # DiffusionLLM-specific parameters
        self.parameters = {
            "model": engine,
            "gen_length": gen_length,
            "temperature": temperature,
            "steps": steps,
            "dual_cache": dual_cache,
            "block_size": block_size,
            "threshold": threshold,
            "return_tokens": return_token
        }

        self.base_url = DLLM_CONFIG["BASE_URL"]
        self.api_key = DLLM_CONFIG["API_KEY"]

        self.chat_url = self.base_url + "generate"
        self.tokenize_url = self.base_url + "tokens"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        logger.info(f"Initialized with engine '{engine}' at {self.base_url}")

    def llm_inference(self, messages):
        """
        Perform DiffusionLLM inference with messages.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            tuple: (response_text, token_count)
        """
        data = self.parameters.copy()
        data['messages'] = messages

        start_time = time.time()
        response = requests.post(url=self.chat_url, headers=self.headers, json=data)
        end_time = time.time()

        if response.status_code != 200:
            raise Exception(f"API_DiffusionLLM inference failed: {response.status_code} - {response.text}")

        response_dict = response.json()

        throughput = response_dict["token"] / (end_time - start_time)
        logger.debug(f"Throughput: {throughput:.2f} tokens/s")

        return (response_dict["response"], response_dict["token"])

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
        data = {
            "model": self.engine,
            "messages": messages
        }

        response = requests.post(self.tokenize_url, headers=self.headers, json=data)

        if response.status_code != 200:
            raise Exception(f"API_DiffusionLLM tokenization failed: {response.status_code} - {response.text}")

        response_dict = response.json()
        return response_dict["num_of_tokens"]

    @classmethod
    def from_config(cls, config):
        """
        Create API_DiffusionLLM instance from configuration dict.

        Args:
            config: Configuration dict with keys:
                - engine (optional, default: "Llada")
                - gen_length (optional, default: 256)
                - steps (optional, default: 256)
                - temperature (optional, default: 0.0)
                - dual_cache (optional, default: False)
                - block_size (optional, default: 32)
                - threshold (optional, default: 0.9)
                - context_length (optional, default: 4000)
                - return_token (optional, default: False)

        Returns:
            API_DiffusionLLM instance
        """
        engine = config.get("engine", "Llada")
        gen_length = config.get("gen_length", 256)
        steps = config.get("steps", 256)
        temperature = config.get("temperature", 0.0)
        dual_cache = config.get("dual_cache", False)
        block_size = config.get("block_size", 32)
        threshold = config.get("threshold", 0.9)
        context_length = config.get("context_length", 4000)
        return_token = config.get("return_token", False)

        return cls(
            engine=engine,
            gen_length=gen_length,
            steps=steps,
            temperature=temperature,
            dual_cache=dual_cache,
            block_size=block_size,
            threshold=threshold,
            context_length=context_length,
            return_token=return_token
        )
