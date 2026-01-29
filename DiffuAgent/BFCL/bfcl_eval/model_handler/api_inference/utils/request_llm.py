import os
import time
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import requests
try:
    from bfcl_eval.model_handler.api_inference.utils.debug_utils import debug
except ImportError:
    debug = None

from types import SimpleNamespace

class REQUEST_LLM:
    def __init__(self,
                 model_path: str="",
                 base_url: str="",
                 api_key: str="",
                 context_length="4096",
                 ):

        self.context_length = context_length
        self.model_path = model_path
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        self.urls = {
            "chat": f"{self.base_url}/v1/chat/completions",
            "model": f"{self.base_url}/models",
            "tokenize": f"{self.base_url}/tokenize",
            "completion": f"{self.base_url}/v1/completions",
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

    def completion(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ):
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        response = requests.post(
            url=self.urls["completion"],
            headers=self.headers,
            json=payload,
            timeout=72000,
        )
        end_time = time.time()

        response.raise_for_status()
        api_response = response.json()
        api_response_object = json.loads(json.dumps(api_response), object_hook=lambda d: SimpleNamespace(**d))

        try:
            text = api_response_object.choices[0].text
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected completion response format: {api_response_object}"
            ) from e

        completion_tokens = api_response_object.usage.completion_tokens

        return SimpleNamespace(
            object=api_response_object,
            json=api_response,
            text=text,
            latency=end_time-start_time,
            num_token=completion_tokens,
        )

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 256,
        quiet: bool = False,
    ):
        """Chat completion with OpenAI-style messages."""

        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dicts")

        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # DEBUG: Log request with nice format
        if not quiet:
            print(f"  ▶ LLM Request: {self.urls['chat']}")
            print(f"    • Model: {self.model_path}")
            print(f"    • Temperature: {temperature}")
            print(f"    • Max tokens: {max_tokens}")

        start_time = time.time()
        response = requests.post(
            url=self.urls["chat"],
            headers=self.headers,
            json=payload,
            timeout=72000,
        )
        end_time = time.time()

        response.raise_for_status()

        api_response = response.json()
        api_response_object = json.loads(json.dumps(api_response), object_hook=lambda d: SimpleNamespace(**d))

        try:
            text = api_response_object.choices[0].message.content
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected chat completion response format: {api_response_object}"
            ) from e

        completion_tokens = api_response_object.usage.completion_tokens
        input_tokens = api_response_object.usage.prompt_tokens

        # DEBUG: Log response with nice format
        if not quiet:
            print(f"  ✓ LLM Response received in {end_time-start_time:.2f}s")
            print(f"    • Status code: {response.status_code}")
            print(f"    • Input tokens: {input_tokens}")
            print(f"    • Output tokens: {completion_tokens}")

        return SimpleNamespace(
            object=api_response_object,
            json=api_response,
            text=text,
            latency=end_time-start_time,
            num_token=completion_tokens,
        )
        
    def chat_completion_with_tools(
        self,
        messages: list[dict],
        tools: list[dict], 
        temperature: float = 0.1,
        max_tokens: int = 256,
    ):
        """Chat completion with OpenAI-style messages."""

        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dicts")

        payload = {
            "model": self.model_path,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto", 
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        response = requests.post(
            url=self.urls["chat"],
            headers=self.headers,
            json=payload,          # ✅ JSON body
            timeout=72000,
        )
        end_time = time.time()

        response.raise_for_status()
        api_response = response.json()
        api_response_object = json.loads(json.dumps(api_response), object_hook=lambda d: SimpleNamespace(**d))

        try:
            text = api_response_object.choices[0].message.content
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected chat completion response format: {api_response_object}"
            ) from e

        completion_tokens = api_response_object.usage.completion_tokens

        return SimpleNamespace(
            object=api_response_object,
            json=api_response,
            text=text,
            latency=end_time-start_time,
            num_token=completion_tokens,
        )


    def num_tokens_from_prompt(self, prompt: str):
        """Return the number of tokens used by a list of messages."""

        payload = {
            "model": self.model_path,
            "prompt": prompt
        }

        response = requests.post(
            url=self.urls["tokenize"], 
            headers=self.headers, 
            json=payload,
            timeout=72000
        )

        response.raise_for_status()
        response_dict = response.json()

        if "count" not in response_dict:
            raise ValueError(
                f"Tokenize API returned unexpected response: {response_dict}"
            )

        return int(response_dict["count"])


    def num_tokens_from_messages(self, messages: list[dict], quiet: bool = False) -> int:
        """Return the number of tokens used by chat-style messages."""

        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dicts")

        payload = {
            "model": self.model_path,
            "messages": messages,
        }

        response = requests.post(
            url=self.urls["tokenize"],
            headers=self.headers,
            json=payload,
            timeout=72000,
        )

        response.raise_for_status()
        response_dict = response.json()

        if "count" not in response_dict:
            raise ValueError(
                f"Tokenize API returned unexpected response: {response_dict}"
            )

        token_count = int(response_dict["count"])

        # DEBUG: Simple one-line output
        if not quiet:
            print(f"[DEBUG] num_tokens_from_messages: model={self.model_path}, tokens={token_count}")

        return token_count

    def num_tokens_from_messages_with_tools(self, messages: list[dict], tools: list[dict]) -> int:
        """Return the number of tokens used by chat-style messages."""

        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dicts")

        payload = {
            "model": self.model_path,
            "messages": messages,
            "tools": tools,
        }

        response = requests.post(
            url=self.urls["tokenize"],
            headers=self.headers,
            json=payload,
            timeout=72000,
        )

        response.raise_for_status()
        response_dict = response.json()

        if "count" not in response_dict:
            raise ValueError(
                f"Tokenize API returned unexpected response: {response_dict}"
            )

        return int(response_dict["count"])