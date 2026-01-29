import os
import time
import json
import requests
from types import SimpleNamespace
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    from bfcl_eval.model_handler.api_inference.utils.debug_utils import debug
except ImportError:
    debug = None

DLLM_TEMPLATE = {
    "Llada": {
        "dual_cache": False,
        "block_size": 32,
        "threshold": 0.9,
        "context_length": 4000,
    },
    "Dream": {
        "dual_cache": True,
        "block_size": 32,
        "threshold": 0.9,
        "context_length": 14000 # test
    },
    "Fdllmv2": {
        "dual_cache": True,
        "block_size": 32,
        "threshold": 0.9,
        "context_length": 32768
    },
    "Wedlm": {
        "dual_cache": True,
        "block_size": 32,
        "threshold": 0.9,
        "context_length": 16384
    },
    "Dllmvar": {
        "dual_cache": True,
        "block_size": 32,
        "threshold": 0.9,
        "context_length": 16384
    },
}

# Default configuration (can be overridden by parameters or environment variables)
# Set via environment variables: DLLM_API_KEY, DLLM_BASE_URL or FEATURES_API_KEY, FEATURES_BASE_URL
API_KEY_DICT = {
    "api_key": os.getenv("DLLM_API_KEY") or os.getenv("FEATURES_API_KEY") or "",
    "base_url": os.getenv("DLLM_BASE_URL") or os.getenv("FEATURES_BASE_URL") or "",
}

class REQUEST_DLLM:
    def __init__(self,
                 model_name: str,
                 base_url: str="",
                 api_key: str="",
                 ):

       
        self.model_name = model_name
        assert model_name in ["Llada", "Dream", "Fdllmv2", "Wedlm", "Dllmvar"]

        self.context_length = DLLM_TEMPLATE[self.model_name]["context_length"]

        self.base_url = base_url if base_url != "" else API_KEY_DICT["base_url"]
        self.api_key = api_key if api_key != "" else API_KEY_DICT["api_key"]

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.urls = {
            "generate": f"{self.base_url}/generate",
            "model": f"{self.base_url}/models",
            "tokenize": f"{self.base_url}/tokens",
        }

    def check_server_availability(self, ) -> None:
        """Check server is ready or not."""

        try:
            # Make a simple request to check if the server is up
            response = requests.get(self.urls["model"], headers=self.headers)
            if response.status_code == 200:
                server_ready = True
                print("server is ready!")
        except:
            raise requests.exceptions.ConnectionError
        
        return

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 256,
        steps: int = -1,
        quiet: bool = False,
    ):
        """Chat completion with OpenAI-style messages."""

        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dicts")

        # Get input token count to safely limit max_tokens
        try:
            input_tokens = self.num_tokens_from_messages(messages, quiet=True)
        except:
            input_tokens = 0

        # Limit max_tokens for DLLM to prevent OOM:
        # 1. Maximum 256 tokens
        # 2. Ensure at least 128 tokens left for safety (input + max_tokens <= context_length - 128)
        safe_max_tokens = min(
            max_tokens,
            256,
            max(0, self.context_length - input_tokens - 128)
        )

        payload = {
            "messages": messages,
            "model": self.model_name,
            "gen_length": safe_max_tokens,
            "temperature": temperature,
            "steps": steps if steps > 0 else safe_max_tokens,
            "dual_cache": DLLM_TEMPLATE[self.model_name]["dual_cache"],
            "block_size": DLLM_TEMPLATE[self.model_name]["block_size"],
            "threshold": DLLM_TEMPLATE[self.model_name]["threshold"],
            "return_tokens": True
        }

        # DEBUG: Log request with nice format
        if not quiet:
            print(f"  ▶ DLLM Request: {self.urls['generate']}")
            print(f"    • Model: {self.model_name}")
            print(f"    • Temperature: {temperature}")
            print(f"    • Max tokens: {max_tokens}")

        start_time = time.time()
        response = requests.post(
            url=self.urls["generate"],
            headers=self.headers,
            json=payload,          # ✅ JSON body
            timeout=60,
        )
        end_time = time.time()

        response.raise_for_status()

        # DEBUG: Log response with nice format
        if not quiet:
            print(f"  ✓ DLLM Response received in {end_time-start_time:.2f}s")
            print(f"    • Status code: {response.status_code}")

        api_response = response.json()
        api_response_object = json.loads(json.dumps(api_response), object_hook=lambda d: SimpleNamespace(**d))

        try:
            text = api_response_object.response
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected chat completion response format: {api_response_object}"
            ) from e

        completion_tokens = api_response_object.token

        # if self.model_name == "WeDLM":
        #     time.sleep(1) # 防止 WeDLM 炸掉

        return SimpleNamespace(
            object=api_response_object,
            json=api_response,
            text=text,
            latency=end_time-start_time,
            num_token=completion_tokens,
        )
        
    def num_tokens_from_messages(self, messages: list[dict], quiet: bool = False) -> int:
        """Return the number of tokens used by chat-style messages."""

        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dicts")

        payload = {
            "model": self.model_name,
            "messages": messages,
        }

        response = requests.post(
            url=self.urls["tokenize"],  # Note: key is "tokenize" but value is "/tokens"
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
        response_dict = response.json()

        if "num_of_tokens" not in response_dict:
            raise ValueError(
                f"Tokenize API returned unexpected response: {response_dict}"
            )

        return int(response_dict["num_of_tokens"])