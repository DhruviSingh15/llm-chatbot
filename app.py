"""
ContextAI - Modern Multi-turn LLM Chatbot with Context Window Management.
Supports Google Gemini & OpenAI with sliding window truncation and running summarization.
"""

import os
import logging
from flask import Flask, jsonify, render_template, request, Response
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llm_client import llm_service, LLMClientError
from context_manager import (
    conversation_manager,
    PERSONAS,
    DEFAULT_PERSONA,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_STRATEGY,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("contextai")

app = Flask(__name__)


# -------------------------------------------------------------
# Frontend Route
# -------------------------------------------------------------
@app.route("/")
def home():
    """Renders the main single-page chat interface."""
    return render_template("index.html")


# -------------------------------------------------------------
# Chat Endpoint (Core Backend API)
# -------------------------------------------------------------
@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main endpoint for user interactions:
    - Receives user message and optional conversation metadata
    - Manages context window (truncation or running summarization)
    - Queries the configured LLM API (Gemini or OpenAI)
    - Returns model response alongside context utilization metrics
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    conversation_id = data.get("conversation_id")
    persona_key = data.get("persona", DEFAULT_PERSONA)
    strategy = data.get("strategy", DEFAULT_STRATEGY)
    max_messages = data.get("max_messages", DEFAULT_MAX_MESSAGES)

    # Validate message input
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a valid message."}), 400

    if len(message) > 8000:
        return jsonify({"error": "Message is too long (maximum 8,000 characters)."}), 400

    # Check if an API key is configured
    if not llm_service.is_configured():
        return jsonify({
            "error": (
                "API key is missing! Please configure GEMINI_API_KEY (Google AI Studio) "
                "or OPENAI_API_KEY in your .env file and restart the server."
            ),
            "configured": False
        }), 503

    try:
        max_messages = int(max_messages)
    except (ValueError, TypeError):
        max_messages = DEFAULT_MAX_MESSAGES

    # Get or initialize conversation session
    conv = conversation_manager.get_or_create(
        conversation_id=conversation_id,
        persona_key=persona_key,
        strategy=strategy,
        max_messages=max_messages
    )
    conversation_id = conv["id"]

    # If settings changed in request, update them
    conversation_manager.update_settings(
        conversation_id=conversation_id,
        persona_key=persona_key,
        strategy=strategy,
        max_messages=max_messages
    )

    # 1. Record user message into multi-turn history
    conversation_manager.add_user_message(conversation_id, message)

    # 2. Context Window Management
    # Prepares sliding window or generates/prepends running summary
    context_data = conversation_manager.prepare_context(conversation_id, llm_service=llm_service)

    try:
        # 3. Call the active LLM Provider
        reply, tokens_used = llm_service.generate_chat_response(
            messages=context_data["context_messages"],
            system_prompt=context_data["effective_system_prompt"],
            temperature=0.7
        )

        if not reply:
            return jsonify({"error": "The AI model returned an empty response. Please retry."}), 502

        # 4. Record assistant message into conversation history
        conversation_manager.add_assistant_message(conversation_id, reply, tokens=tokens_used)

        # 5. Return response with context metrics for UI inspection
        return jsonify({
            "conversation_id": conversation_id,
            "reply": reply,
            "total_messages": context_data["total_messages"] + 1,  # including assistant reply
            "context_messages_count": context_data["context_messages_count"],
            "truncated_count": context_data["truncated_count"],
            "context_limit": conv["max_messages"],
            "strategy": conv["strategy"],
            "has_summary": context_data["has_summary"],
            "running_summary": context_data.get("running_summary", ""),
            "tokens": tokens_used,
            "provider": llm_service.provider,
            "model": llm_service.model
        })

    except LLMClientError as exc:
        logger.warning(f"LLMClientError on /chat: {exc.message}")
        return jsonify({"error": exc.message}), exc.status_code

    except Exception as exc:
        logger.exception("Unhandled server exception in /chat")
        return jsonify({
            "error": f"An unexpected error occurred: {str(exc)}. Please try again."
        }), 500


# -------------------------------------------------------------
# Configuration & Provider Status
# -------------------------------------------------------------
@app.route("/api/config", methods=["GET"])
def get_config():
    """Returns application configuration, active provider, and available options."""
    info = llm_service.get_info()
    return jsonify({
        "configured": info["configured"],
        "provider": info["provider"],
        "model": info["model"],
        "personas": PERSONAS,
        "default_persona": DEFAULT_PERSONA,
        "strategies": [
            {
                "id": "sliding_window",
                "name": "Sliding Window Truncation",
                "description": "Retains the latest N messages; drops older messages when window overflows."
            },
            {
                "id": "summarization",
                "name": "Running Summarization",
                "description": "Summarizes older messages into a rolling recap prepended to system instructions."
            }
        ],
        "default_strategy": DEFAULT_STRATEGY,
        "default_max_messages": DEFAULT_MAX_MESSAGES
    })


# -------------------------------------------------------------
# Conversation Management Endpoints
# -------------------------------------------------------------
@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    """Returns a list of active conversations."""
    return jsonify({"conversations": conversation_manager.list_conversations()})


@app.route("/api/conversations/new", methods=["POST"])
def new_conversation():
    """Explicitly initializes a new conversation session."""
    data = request.get_json(silent=True) or {}
    conv = conversation_manager.get_or_create(
        conversation_id=None,
        persona_key=data.get("persona", DEFAULT_PERSONA),
        strategy=data.get("strategy", DEFAULT_STRATEGY),
        max_messages=data.get("max_messages", DEFAULT_MAX_MESSAGES)
    )
    return jsonify({"conversation": conv})


@app.route("/api/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    """Retrieves full conversation details, history, and context metrics."""
    with conversation_manager._lock:
        conv = conversation_manager._conversations.get(conversation_id)
        if not conv:
            return jsonify({"error": "Conversation not found."}), 404
        return jsonify({"conversation": conv})


@app.route("/api/conversations/<conversation_id>/clear", methods=["POST"])
def clear_conversation(conversation_id):
    """Clears history for a specific conversation."""
    success = conversation_manager.clear_conversation(conversation_id)
    if not success:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"status": "cleared", "conversation_id": conversation_id})


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id):
    """Deletes an entire conversation."""
    success = conversation_manager.delete_conversation(conversation_id)
    if not success:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"status": "deleted", "conversation_id": conversation_id})


@app.route("/api/conversations/<conversation_id>/export", methods=["GET"])
def export_conversation(conversation_id):
    """Exports conversation transcript as Markdown or JSON."""
    fmt = request.args.get("format", "markdown").lower()
    content = conversation_manager.export_conversation(conversation_id, fmt=fmt)
    if not content:
        return jsonify({"error": "Conversation not found or empty."}), 404

    mime = "application/json" if fmt == "json" else "text/markdown; charset=utf-8"
    filename = f"conversation_{conversation_id[:8]}.{fmt}"
    return Response(
        content,
        mimetype=mime,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# -------------------------------------------------------------
# Health Check
# -------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    info = llm_service.get_info()
    return jsonify({
        "status": "healthy",
        "configured": info["configured"],
        "provider": info["provider"],
        "model": info["model"]
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n=======================================================")
    print(f" ContextAI Chatbot is running at http://127.0.0.1:{port}")
    info = llm_service.get_info()
    if info["configured"]:
        print(f" Active Provider: {info['provider'].upper()} (model: {info['model']})")
    else:
        print(" [!] No API key configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
