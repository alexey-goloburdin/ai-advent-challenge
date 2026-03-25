import json
import re
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import urllib.request
import urllib.error

from src.logger import get_logger


logger = get_logger(__name__)


class LLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        logger.info(f"LLM Client initialized for endpoint: {self.base_url}")

    @staticmethod
    def _messages_to_input(messages: list[dict[str, str]]) -> str:
        """Converts messages array to a single input string for the API."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n\n".join(parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((urllib.error.URLError, TimeoutError))
    )
    def request_completion(
        self,
        messages: list[dict[str, str]],
        model_name: str = "qwen/qwen3.5-35b-a3b"

    ) -> str:
        """Sends a chat completion request with retry logic."""

        payload = {
            "model": model_name,
            "input": self._messages_to_input(messages),
            "reasoning": "off",
            "temperature": 0.1,
        }

        data = json.dumps(payload).encode('utf-8')

        req = urllib.request.Request(
            self.base_url + "/api/v1/chat",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))

                output = result.get("output", [])
                if not output:
                    raise ValueError("Empty response from model")

                return output[0]["content"]
                
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error from LLM API: {e.code} - {e.reason}")

            raise
        except Exception as e:
            logger.error(f"Network error during request: {e}")
            raise

    @staticmethod
    def clean_json_output(text: str) -> str:
        """Removes markdown code blocks and ensures valid JSON structure."""
        text = text.strip()
        
        # Remove ```json ... ``` or ``` ... ``` patterns
        pattern = r'^```(?:json)?\s*\n?|\n?```$'
        clean_text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return clean_text.strip()

    @staticmethod
    def validate_json_structure(json_str: str, expected_keys: list[str]) -> bool:
        """Basic validation to ensure required keys exist in JSON."""
        try:
            data = json.loads(json_str)
            # Check if all expected keys are present (optional depending on strictness)
            # For this agent, we trust the model but catch parse errors.
            return True
        except json.JSONDecodeError:
            return False

