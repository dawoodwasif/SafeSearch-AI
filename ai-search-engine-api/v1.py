import requests
import json
from typing import Generator
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

class v1:
    """
    A class to interact with the v1 AI API.
    """

    AVAILABLE_MODELS = ["llama", "claude"]

    def __init__(
        self,
        model: str = "claude",
        timeout: int = 300,
        proxies: dict = {},
    ):
        """
        Initializes the v1 AI API with given parameters.
        Args:
            model (str, optional): The AI model to use for text generation. Defaults to "claude". 
                                    Options: "llama", "claude".
            timeout (int, optional): Http request timeout. Defaults to 30.
            proxies (dict, optional): Http request proxies. Defaults to {}.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model}' is not supported. Choose from {self.AVAILABLE_MODELS}.")

        self.session = requests.Session()
        self.api_endpoint = os.getenv("API_ENDPOINT")
        self.timeout = timeout
        self.model = model
        self.device_token = self.get_device_token()

        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
        )
        self.session.proxies = proxies

    def get_device_token(self) -> str:
        device_token_url = os.getenv("DEVICE_TOKEN_URL")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        data = {}
        response = requests.post(
            device_token_url, headers=headers, data=json.dumps(data)
        )

        if response.status_code == 200:
            device_token_data = response.json()
            return device_token_data["sessionToken"]
        else:
            raise Exception(
                f"Failed to get device token - ({response.status_code}, {response.reason}) - {response.text}"
            )

    def ask(self, prompt: str) -> Generator[str, None, None]:
        search_data = {"query": prompt, "deviceToken": self.device_token}

        response = self.session.post(
            self.api_endpoint, json=search_data, stream=True, timeout=self.timeout
        )
        if not response.ok:
            raise Exception(
                f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
            )

        buffer = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if data['type'] == 'chunk':
                            model = data['model']
                            if (self.model == "llama" and model == 'OPENROUTER_LLAMA_3') or \
                               (self.model == "claude" and model == 'OPENROUTER_CLAUDE'):
                                content = data['chunk']['content']
                                if content:
                                    buffer += content
                                    # Check if we have a complete line or paragraph
                                    lines = buffer.split('\n')
                                    if len(lines) > 1:
                                        for complete_line in lines[:-1]:
                                            yield self.format_text(complete_line) + '\n'
                                        buffer = lines[-1]
                    except KeyError:
                        pass
                    except json.JSONDecodeError:
                        pass

        # Yield any remaining content in the buffer
        if buffer:
            yield self.format_text(buffer)

        yield "[DONE]"

    def format_text(self, text: str) -> str:
        # Convert *text* to <i>text</i> for italic
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        return text

    def chat(self, prompt: str) -> Generator[str, None, None]:
        """Stream responses as string chunks"""
        return self.ask(prompt)