"""
LLM Client Abstraction for ContextAI.
Supports both OpenAI and Google Gemini with automatic provider detection,
unified message formatting, and resilient error handling.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Custom exception for LLM errors with HTTP status codes and user-facing messages."""
    def __init__(self, message: str, status_code: int = 502, raw_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.raw_error = raw_error


class LLMService:
    def __init__(self):
        self.provider = None
        self.model = None
        self.openai_client = None
        self.gemini_client = None
        self._init_provider()

    def _init_provider(self):
        """Detect and initialize the active LLM provider (Gemini or OpenAI)."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        forced_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Select provider based on preference or available keys
        if forced_provider == "gemini" and gemini_key:
            self._setup_gemini(gemini_key)
        elif forced_provider == "openai" and openai_key:
            self._setup_openai(openai_key)
        elif gemini_key:
            # Default to Gemini if key is present (free tier friendly)
            self._setup_gemini(gemini_key)
        elif openai_key:
            self._setup_openai(openai_key)
        else:
            self.provider = None
            self.model = None
            logger.warning("No valid API key found for OpenAI or Gemini.")

    def _setup_gemini(self, api_key: str):
        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=api_key)
            self.provider = "gemini"
            # Default to gemini-2.0-flash or gemini-1.5-flash
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            logger.info(f"Gemini client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.gemini_client = None
            self.provider = None

    def _setup_openai(self, api_key: str):
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=api_key,
                timeout=30.0,
                max_retries=2
            )
            self.provider = "openai"
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"OpenAI client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
            self.provider = None

    def is_configured(self) -> bool:
        """Returns True if a provider client is ready to make requests."""
        # Re-check in case .env was modified after startup
        if not self.provider:
            self._init_provider()
        return self.provider is not None

    def get_info(self) -> Dict[str, Any]:
        """Get summary info about the active provider."""
        return {
            "configured": self.is_configured(),
            "provider": self.provider or "none",
            "model": self.model or "none",
        }

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7
    ) -> Tuple[str, int]:
        """
        Sends formatted message history to the active LLM provider.
        Returns: (assistant_reply_text, estimated_or_actual_tokens)
        """
        if not self.is_configured():
            raise LLMClientError(
                "API key is not configured. Please add GEMINI_API_KEY or OPENAI_API_KEY to your .env file.",
                status_code=503
            )

        if self.provider == "gemini":
            return self._call_gemini(messages, system_prompt, temperature)
        elif self.provider == "openai":
            return self._call_openai(messages, system_prompt, temperature)
        else:
            raise LLMClientError("Unsupported LLM provider.", status_code=500)

    def generate_summary(self, conversation_text: str) -> str:
        """Generates a concise running summary of older messages."""
        if not self.is_configured():
            return "Prior conversation context preserved."

        prompt = (
            "Summarize the key facts, user preferences, technical details, and context "
            "from the following conversation transcript in 2-4 concise bullet points so "
            "future turns understand the full context:\n\n"
            f"{conversation_text}\n\n"
            "Summary:"
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            reply, _ = self.generate_chat_response(
                messages=messages,
                system_prompt="You are an expert conversation summarizer. Be concise and capture vital context.",
                temperature=0.3
            )
            return reply.strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}. Falling back to default note.")
            return "Prior messages summarized for context retention."

    # ------------------ OpenAI Provider Implementation ------------------
    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float
    ) -> Tuple[str, int]:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            AuthenticationError,
            APIStatusError
        )

        formatted_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            formatted_messages.append({
                "role": m.get("role", "user"),
                "content": m.get("content", "")
            })

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature
            )

            reply = response.choices[0].message.content or ""
            tokens = getattr(response.usage, "total_tokens", len(reply) // 4)
            return reply, tokens

        except AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}")
            raise LLMClientError(
                "Invalid OpenAI API key. Please check your OPENAI_API_KEY in .env.",
                status_code=401,
                raw_error=e
            )
        except RateLimitError as e:
            logger.warning(f"OpenAI Rate Limit: {e}")
            raise LLMClientError(
                "OpenAI rate limit or quota exceeded. Check your plan at platform.openai.com.",
                status_code=429,
                raw_error=e
            )
        except APITimeoutError as e:
            logger.error(f"OpenAI Timeout: {e}")
            raise LLMClientError(
                "OpenAI request timed out. Please try your message again.",
                status_code=504,
                raw_error=e
            )
        except APIConnectionError as e:
            logger.error(f"OpenAI Connection Error: {e}")
            raise LLMClientError(
                "Could not connect to OpenAI. Please verify your internet connection.",
                status_code=502,
                raw_error=e
            )
        except APIStatusError as e:
            logger.error(f"OpenAI API Status Error: {e}")
            raise LLMClientError(
                f"OpenAI returned an error (status {e.status_code}). Please try again shortly.",
                status_code=502,
                raw_error=e
            )
        except Exception as e:
            logger.exception("Unexpected OpenAI failure")
            raise LLMClientError(
                f"An unexpected error occurred while communicating with OpenAI: {str(e)}",
                status_code=500,
                raw_error=e
            )

    # ------------------ Gemini Provider Implementation ------------------
    def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float
    ) -> Tuple[str, int]:
        import time
        from google.genai import types
        from google.genai.errors import APIError

        contents = []
        for m in messages:
            role = "user" if m.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m.get("content", ""))]
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature
        )

        # Try active model, with fallback models and retry on temporary 503/demand spikes
        candidate_models = list(dict.fromkeys([self.model, "gemini-3.5-flash-lite", "gemini-3.5-flash"]))
        last_error = None

        for attempt, mod in enumerate(candidate_models):
            try:
                response = self.gemini_client.models.generate_content(
                    model=mod,
                    contents=contents,
                    config=config
                )

                reply = response.text or ""
                tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    tokens = getattr(response.usage_metadata, "total_token_count", 0)
                if not tokens:
                    tokens = (len(system_prompt) + sum(len(m.get("content", "")) for m in messages) + len(reply)) // 4

                return reply, tokens

            except (APIError, Exception) as e:
                last_error = e
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    raise LLMClientError(
                        "Gemini API rate limit reached. Please wait a moment before trying again.",
                        status_code=429,
                        raw_error=e
                    )
                elif "INVALID_ARGUMENT" in msg or "API_KEY_INVALID" in msg:
                    raise LLMClientError(
                        "Invalid Gemini API key. Please check your GEMINI_API_KEY in .env.",
                        status_code=401,
                        raw_error=e
                    )
                elif "503" in msg or "high demand" in msg.lower() or "disconnected" in msg.lower() or "connection" in msg.lower():
                    logger.warning(f"Gemini {mod} transient network/server glitch ({msg}). Retrying in 1.5s... (attempt {attempt+1})")
                    time.sleep(1.5)
                    continue
                else:
                    logger.exception(f"Gemini API failure: {msg}")
                    raise LLMClientError(
                        f"Gemini API error: {e.message if hasattr(e, 'message') else str(e)}",
                        status_code=502,
                        raw_error=e
                    )

        # If all candidates exhausted
        raise LLMClientError(
            f"Gemini API is currently experiencing high demand. Please try again in a few moments.",
            status_code=503,
            raw_error=last_error
        )


# Global singleton instance
llm_service = LLMService()
