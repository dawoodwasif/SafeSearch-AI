import requests
import json
from typing import Generator, Optional
import os
from dotenv import load_dotenv
import re
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    user_prompt: str
    system_prompt: Optional[str] = "You are a helpful AI assistant."

class CHATv1:
    """
    A class to interact with the CHATv1.info API.
    """

    def __init__(
        self,
        timeout: int = 300,
        proxies: dict = {},
    ):
        """
        Initializes the CHATv1.info API with given parameters.
        
        Args:
            timeout (int, optional): Http request timeout. Defaults to 300.
            proxies (dict, optional): Http request proxies. Defaults to {}.
        """
        self.session = requests.Session()
        self.api_endpoint = os.getenv("CHATv1")
        self.timeout = timeout
        self.headers = {
            "content-type": "application/json",
        }
        self.session.headers.update(self.headers)
        self.session.proxies = proxies

    def ask(self, user_prompt: str, system_prompt: str) -> Generator[str, None, None]:
        """
        Chat with AI

        Args:
            user_prompt (str): User's prompt to be sent.
            system_prompt (str): System prompt to set the AI's behavior.

        Yields:
            str: Incremental text responses.
        """
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        response = self.session.post(self.api_endpoint, json=payload, stream=True, timeout=self.timeout)
        
        if not response.ok:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
            )

        buffer = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data_json = json.loads(data_str)
                        content = data_json.get("data", "")
                        if content:
                            buffer += content
                            lines = buffer.split('\n')
                            if len(lines) > 1:
                                for complete_line in lines[:-1]:
                                    yield self.format_text(complete_line) + '\n'
                                buffer = lines[-1]
                    except json.JSONDecodeError:
                        pass

        if buffer:
            yield self.format_text(buffer)
        
        yield "[DONE]"

    def format_text(self, text: str) -> str:
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        return text

    def chat(self, user_prompt: str, system_prompt: str) -> Generator[str, None, None]:
        """Stream responses as string chunks"""
        return self.ask(user_prompt, system_prompt)


