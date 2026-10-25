"""
Test Suite for ContextAI.
Validates:
1. In-memory multi-turn conversation persistence
2. Context window truncation strategy (sliding window)
3. Running summarization strategy
4. Backend API endpoint (/chat, /api/config, /health)
5. Robust error handling (missing API key, rate limits, empty messages)
"""

import unittest
from unittest.mock import patch, MagicMock
from app import app
from context_manager import ConversationManager, PERSONAS
from llm_client import LLMClientError


class TestContextManagement(unittest.TestCase):
    def setUp(self):
        self.cm = ConversationManager()

    def test_single_turn_addition(self):
        conv = self.cm.get_or_create()
        cid = conv["id"]
        self.cm.add_user_message(cid, "Hello! My name is Alex.")
        self.cm.add_assistant_message(cid, "Hi Alex! Nice to meet you.")

        ctx = self.cm.prepare_context(cid)
        self.assertEqual(ctx["total_messages"], 2)
        self.assertEqual(ctx["context_messages_count"], 2)
        self.assertEqual(ctx["truncated_count"], 0)
        self.assertEqual(ctx["context_messages"][0]["content"], "Hello! My name is Alex.")

    def test_multi_turn_history_persistence(self):
        conv = self.cm.get_or_create()
        cid = conv["id"]

        # Turn 1
        self.cm.add_user_message(cid, "I am studying Quantum Computing.")
        self.cm.add_assistant_message(cid, "Great subject! What aspect?")

        # Turn 2
        self.cm.add_user_message(cid, "Specifically superposition and qubits.")
        self.cm.add_assistant_message(cid, "Superposition allows qubits to exist in linear combinations.")

        # Turn 3
        self.cm.add_user_message(cid, "What was the first subject I mentioned?")

        ctx = self.cm.prepare_context(cid)
        self.assertEqual(ctx["total_messages"], 5)
        # Verify earlier message is still present in context for the model to reference
        self.assertIn("Quantum Computing", ctx["context_messages"][0]["content"])

    def test_sliding_window_truncation(self):
        """Verify that when message count exceeds max_messages, only the latest N are kept."""
        limit = 6
        conv = self.cm.get_or_create(strategy="sliding_window", max_messages=limit)
        cid = conv["id"]

        # Add 12 messages (6 turns)
        for i in range(6):
            self.cm.add_user_message(cid, f"User message #{i}")
            self.cm.add_assistant_message(cid, f"Assistant reply #{i}")

        ctx = self.cm.prepare_context(cid)
        self.assertEqual(ctx["total_messages"], 12)
        self.assertEqual(ctx["context_messages_count"], limit)
        self.assertEqual(ctx["truncated_count"], 6)

        # The oldest message in context must now be User message #3 (the latest 6 messages are #3 to #5)
        self.assertEqual(ctx["context_messages"][0]["content"], "User message #3")
        self.assertEqual(ctx["context_messages"][-1]["content"], "Assistant reply #5")

    def test_running_summarization_strategy(self):
        """Verify summarization mode generates and prepends rolling recap."""
        limit = 4
        mock_llm = MagicMock()
        mock_llm.generate_summary.return_value = "User is building a web app in Python."

        conv = self.cm.get_or_create(strategy="summarization", max_messages=limit)
        cid = conv["id"]

        # Add 6 messages
        for i in range(3):
            self.cm.add_user_message(cid, f"Building step {i}")
            self.cm.add_assistant_message(cid, f"Acknowledged step {i}")

        ctx = self.cm.prepare_context(cid, llm_service=mock_llm)
        self.assertEqual(ctx["total_messages"], 6)
        self.assertEqual(ctx["truncated_count"], 2)
        self.assertTrue(ctx["has_summary"])
        self.assertIn("RECAP OF EARLIER CONVERSATION", ctx["effective_system_prompt"])
        self.assertIn("User is building a web app in Python.", ctx["effective_system_prompt"])

    def test_persona_system_prompt(self):
        conv = self.cm.get_or_create(persona_key="architect")
        cid = conv["id"]
        ctx = self.cm.prepare_context(cid)
        self.assertIn("Senior Software Architect", ctx["effective_system_prompt"])


class TestChatbotAPI(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_config_endpoint(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("personas", data)
        self.assertIn("strategies", data)

    def test_empty_message_validation(self):
        response = self.client.post("/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    @patch("app.llm_service.is_configured", return_value=False)
    def test_missing_api_key_error_handling(self, mock_is_configured):
        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertIn("API key is missing", data["error"])

    @patch("app.llm_service.is_configured", return_value=True)
    @patch("app.llm_service.generate_chat_response")
    def test_successful_chat_turn(self, mock_generate, mock_is_configured):
        mock_generate.return_value = ("Hello! How can I assist you with your studies?", 25)
        response = self.client.post("/chat", json={"message": "Hello, ContextAI!"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("conversation_id", data)
        self.assertEqual(data["reply"], "Hello! How can I assist you with your studies?")
        self.assertEqual(data["total_messages"], 2)
        self.assertEqual(data["context_messages_count"], 1)

    @patch("app.llm_service.is_configured", return_value=True)
    @patch("app.llm_service.generate_chat_response")
    def test_api_rate_limit_error_handling(self, mock_generate, mock_is_configured):
        mock_generate.side_effect = LLMClientError(
            "Rate limit reached. Please wait before retrying.", status_code=429
        )
        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 429)
        data = response.get_json()
        self.assertIn("Rate limit reached", data["error"])


if __name__ == "__main__":
    unittest.main()
