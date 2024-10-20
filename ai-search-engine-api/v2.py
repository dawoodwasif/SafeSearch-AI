import re
import requests
from uuid import uuid4
import json
from typing import Any, AsyncGenerator, Dict, Generator
import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class v2:
    def __init__(
        self,
        timeout: int = 100,
        proxies: dict = {},
    ):  
        self.session = requests.Session()
        self.chat_endpoint = os.getenv("v2")
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
        }

        self.session.headers.update(self.headers)
        self.session.proxies = proxies

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        conversation_prompt = prompt

        self.session.headers.update(self.headers)
        payload = {
            "query": conversation_prompt,
            "search_uuid": uuid4().hex,
            "lang": "",
            "agent_lang": "en",
            "search_options": {
                "langcode": "en-US"
            },
            "search_video": True,
            "contexts_from": "google"
        }

        response = self.session.post(
            self.chat_endpoint, json=payload, stream=True, timeout=self.timeout
        )
        if not response.ok:
            raise Exception(
                f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
            )

        streaming_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data:'):
                try:
                    data = json.loads(line[5:].strip())
                    if data['type'] == 'answer' and 'text' in data['data']:
                        new_text = data['data']['text']
                        if len(new_text) > len(streaming_text):
                            delta = new_text[len(streaming_text):]
                            streaming_text = new_text
                            resp = dict(text=delta)
                            self.last_response.update(dict(text=streaming_text))
                            yield line if raw else resp
                except json.JSONDecodeError:
                    pass

    def chat(
        self,
        prompt: str,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        buffer = ""
        for response in self.ask(prompt, True):
            text = self.get_message(response)
            buffer += text
            lines = buffer.split('\n')
            for line in lines[:-1]:
                yield self.format_text(line) + '\n\n'
            buffer = lines[-1]
        if buffer:
            yield self.format_text(buffer) + '\n\n'
        yield "[DONE]"

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        
        if "text" in response:
            text = re.sub(r'\[\[\d+\]\]', '', response["text"])
            return text
        else:
            return ""

    def format_text(self, text: str) -> str:
        # Convert *text* to <i>text</i> for italic
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        return text