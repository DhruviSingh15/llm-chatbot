"""
Context Window & Conversation Manager for ContextAI.

Provides:
- In-memory conversation persistence per conversation_id
- Sliding window truncation (last N messages)
- Running conversation summarization (stretch goal)
- Persona & system prompt configurations
- Real-time token estimation
- Thread-safe synchronization
"""

import time
import uuid
from threading import Lock
from typing import Dict, List, Any, Optional

# Predefined personas giving ContextAI a defined role, personality, and constraints
PERSONAS = {
    "mentor": {
        "name": "ContextAI Mentor",
        "description": "Friendly, insightful study and technical mentor",
        "system_prompt": (
            "You are ContextAI, a friendly, encouraging, and highly competent study and technical mentor. "
            "You explain complex ideas step-by-step with clear analogies and real-world examples. "
            "Encourage deep understanding, format code with syntax highlighting, and proactively "
            "ask clarifying questions when requirements are ambiguous. Always maintain context across all turns."
        )
    },
    "architect": {
        "name": "Code Architect",
        "description": "Senior software engineer focused on clean architecture & best practices",
        "system_prompt": (
            "You are ContextAI acting as a pragmatic Senior Software Architect. "
            "Prioritize clean architecture, design patterns, security, performance, and maintainability. "
            "Provide production-grade code snippets, call out edge cases, and give direct, actionable technical feedback."
        )
    },
    "socratic": {
        "name": "Socratic Tutor",
        "description": "Guides you to answers by asking thought-provoking questions",
        "system_prompt": (
            "You are ContextAI in Socratic Tutor mode. "
            "Instead of just handing over the answer, guide the user to discover solutions on their own "
            "through targeted, thought-provoking questions, hints, and step-by-step reasoning."
        )
    },
    "concise": {
        "name": "Executive Brief",
        "description": "Brief, bulleted, high-impact responses with zero fluff",
        "system_prompt": (
            "You are ContextAI in Executive mode. "
            "Provide concise, high-signal, bulleted answers. Omit pleasantries, dive straight into key takeaways, "
            "and emphasize high-leverage insights and action items."
        )
    }
}

DEFAULT_PERSONA = "mentor"
DEFAULT_MAX_MESSAGES = 10
DEFAULT_STRATEGY = "sliding_window"  # "sliding_window" or "summarization"


class ConversationManager:
    def __init__(self):
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get_or_create(
        self,
        conversation_id: Optional[str] = None,
        persona_key: str = DEFAULT_PERSONA,
        strategy: str = DEFAULT_STRATEGY,
        max_messages: int = DEFAULT_MAX_MESSAGES
    ) -> Dict[str, Any]:
        """Retrieve an existing conversation or initialize a new one."""
        with self._lock:
            if not conversation_id or conversation_id not in self._conversations:
                conversation_id = str(uuid.uuid4())
                persona = PERSONAS.get(persona_key, PERSONAS[DEFAULT_PERSONA])
                self._conversations[conversation_id] = {
                    "id": conversation_id,
                    "title": "New Conversation",
                    "persona_key": persona_key if persona_key in PERSONAS else DEFAULT_PERSONA,
                    "system_prompt": persona["system_prompt"],
                    "strategy": strategy if strategy in ("sliding_window", "summarization") else DEFAULT_STRATEGY,
                    "max_messages": max(2, min(max_messages, 50)),
                    "messages": [],           # All messages in the conversation
                    "running_summary": "",    # Pre-calculated summary for older messages
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "total_turns": 0
                }
            return self._conversations[conversation_id]

    def add_user_message(self, conversation_id: str, content: str) -> Dict[str, Any]:
        """Appends a user message and returns the conversation."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                conv = self.get_or_create(conversation_id)

            # Auto-title on first message
            if not conv["messages"]:
                snippet = content.strip().split("\n")[0][:40]
                conv["title"] = snippet if len(snippet) < 37 else snippet[:37] + "..."

            msg = {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": content.strip(),
                "timestamp": time.time(),
                "tokens": self.estimate_tokens(content)
            }
            conv["messages"].append(msg)
            conv["updated_at"] = time.time()
            return conv

    def add_assistant_message(self, conversation_id: str, content: str, tokens: int = 0) -> Dict[str, Any]:
        """Appends an assistant message and updates turn counts."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                return {}

            msg = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": content,
                "timestamp": time.time(),
                "tokens": tokens or self.estimate_tokens(content)
            }
            conv["messages"].append(msg)
            conv["total_turns"] += 1
            conv["updated_at"] = time.time()
            return conv

    def prepare_context(self, conversation_id: str, llm_service=None) -> Dict[str, Any]:
        """
        Executes Context Window Management strategy:
        1. Sliding Window: Truncates history to the last `max_messages`.
        2. Summarization: Truncates to last `max_messages`, while maintaining
           a running summary of older truncated messages and injecting it into the system prompt.

        Returns:
            {
                "context_messages": list of messages sent to LLM,
                "effective_system_prompt": system prompt with running summary if applicable,
                "total_messages": int,
                "context_messages_count": int,
                "truncated_count": int,
                "has_summary": bool,
                "strategy": str
            }
        """
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                return {
                    "context_messages": [],
                    "effective_system_prompt": PERSONAS[DEFAULT_PERSONA]["system_prompt"],
                    "total_messages": 0,
                    "context_messages_count": 0,
                    "truncated_count": 0,
                    "has_summary": False,
                    "strategy": DEFAULT_STRATEGY
                }

            all_messages = conv["messages"]
            limit = conv["max_messages"]
            strategy = conv["strategy"]
            system_prompt = conv["system_prompt"]
            running_summary = conv.get("running_summary", "")

        total = len(all_messages)

        if total <= limit:
            # Entire conversation fits within context window
            context_msgs = [{"role": m["role"], "content": m["content"]} for m in all_messages]
            return {
                "context_messages": context_msgs,
                "effective_system_prompt": system_prompt,
                "total_messages": total,
                "context_messages_count": total,
                "truncated_count": 0,
                "has_summary": bool(running_summary),
                "strategy": strategy
            }

        # Context exceeds limit - perform context window management
        truncated_count = total - limit
        recent_messages = all_messages[-limit:]
        context_msgs = [{"role": m["role"], "content": m["content"]} for m in recent_messages]

        effective_system_prompt = system_prompt

        if strategy == "summarization":
            # If summarization strategy is chosen and we have older messages:
            older_messages = all_messages[:-limit]
            # Create transcript snippet of older messages
            transcript_lines = []
            for m in older_messages[-8:]:  # summarize the preceding overflow chunk
                transcript_lines.append(f"{m['role'].upper()}: {m['content']}")
            older_transcript = "\n".join(transcript_lines)

            # Generate or update summary if LLM service is available and not already updated
            if llm_service and (not running_summary or len(older_messages) % 4 == 0):
                try:
                    summary_update = llm_service.generate_summary(older_transcript)
                    with self._lock:
                        conv["running_summary"] = summary_update
                        running_summary = summary_update
                except Exception:
                    pass

            if running_summary:
                effective_system_prompt = (
                    f"{system_prompt}\n\n"
                    f"--- RECAP OF EARLIER CONVERSATION (Context Retention) ---\n"
                    f"{running_summary}\n"
                    f"--- END OF RECAP ---\n"
                    f"Use the recap above when the user refers to earlier parts of the discussion."
                )

        return {
            "context_messages": context_msgs,
            "effective_system_prompt": effective_system_prompt,
            "total_messages": total,
            "context_messages_count": len(context_msgs),
            "truncated_count": truncated_count,
            "has_summary": bool(running_summary),
            "running_summary": running_summary,
            "strategy": strategy
        }

    def update_settings(
        self,
        conversation_id: str,
        persona_key: Optional[str] = None,
        strategy: Optional[str] = None,
        max_messages: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update persona, strategy, or limit for a conversation."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                return {}

            if persona_key and persona_key in PERSONAS:
                conv["persona_key"] = persona_key
                conv["system_prompt"] = PERSONAS[persona_key]["system_prompt"]

            if strategy and strategy in ("sliding_window", "summarization"):
                conv["strategy"] = strategy

            if max_messages and 2 <= max_messages <= 50:
                conv["max_messages"] = max_messages

            return conv

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clears all messages in an existing conversation while keeping its settings."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                return False
            conv["messages"] = []
            conv["running_summary"] = ""
            conv["total_turns"] = 0
            conv["updated_at"] = time.time()
            return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Deletes the conversation completely."""
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                return True
            return False

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Returns metadata for all conversations sorted by last update."""
        with self._lock:
            summaries = []
            for cid, conv in self._conversations.items():
                summaries.append({
                    "id": cid,
                    "title": conv["title"],
                    "persona_key": conv["persona_key"],
                    "strategy": conv["strategy"],
                    "message_count": len(conv["messages"]),
                    "total_turns": conv["total_turns"],
                    "updated_at": conv["updated_at"],
                    "created_at": conv["created_at"]
                })
            summaries.sort(key=lambda x: x["updated_at"], reverse=True)
            return summaries

    def export_conversation(self, conversation_id: str, fmt: str = "markdown") -> str:
        """Exports the conversation history in Markdown or JSON format."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                return ""

        if fmt == "json":
            import json
            return json.dumps(conv, indent=2)

        # Markdown format
        persona_name = PERSONAS.get(conv["persona_key"], {}).get("name", "ContextAI")
        lines = [
            f"# ContextAI Conversation: {conv['title']}",
            f"- **Persona**: {persona_name}",
            f"- **Strategy**: {conv['strategy']} (Limit: {conv['max_messages']} msgs)",
            f"- **Total Turns**: {conv['total_turns']}",
            ""
        ]

        if conv.get("running_summary"):
            lines.extend([
                "## Running Memory Summary",
                conv["running_summary"],
                ""
            ])

        lines.append("## Transcript")
        for m in conv["messages"]:
            speaker = "You" if m["role"] == "user" else "ContextAI"
            lines.append(f"### {speaker}")
            lines.append(m["content"])
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimation (~4 characters per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)


# Global singleton instance
conversation_manager = ConversationManager()
