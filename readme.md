# ✦ ContextAI — Intelligent Multi-Turn LLM Chatbot

> A production-grade multi-turn conversational AI backed by real LLM APIs (**Google Gemini** & **OpenAI**) featuring **sliding window context truncation**, **running summarization**, customizable personas, and a sleek modern web interface and CLI.

---

## 🎯 Architecture & Highlights

ContextAI solves the core challenge of LLM conversational agents: **context window management**. Raw APIs forget prior turns immediately, while sending unlimited history quickly exhausts token limits and increases latency. ContextAI delivers:

- 🧠 **Multi-Turn Context Persistence**: Messages are saved per session, enabling the assistant to effortlessly reference statements made earlier in the dialogue.
- ⚡ **Dual Context Window Strategies**:
  1. **Sliding Window (Truncation)**: Retains the latest $N$ messages (configurable 4–24 messages), guaranteeing prompt bounds while maintaining immediate conversational context.
  2. **Running Summarization (Stretch Goal)**: When the conversation overflows the context window, older turns are distilled into a rolling summary and injected into the system prompt.
- 🎭 **Curated System Prompts & Personas**: Pre-configured identities (*Technical Mentor*, *Code Architect*, *Socratic Tutor*, and *Executive Brief*) with constraints and role behavior.
- 🛡️ **Graceful Error Handling**: Maps rate limits (429), expired/invalid keys (401), timeouts (504), network disconnects (502), and invalid payloads (400) to clear, friendly user notifications without breaking the UI.
- 🔌 **Multi-Provider Support**: Seamlessly works with **Google Gemini** (`gemini-2.0-flash`, `gemini-1.5-flash`) and **OpenAI** (`gpt-4o-mini`, `gpt-4o`) with automatic environment detection.
- 🎨 **State-of-the-Art Interface**: Glassmorphism, dark/light theme, live **Context Window Inspector**, code syntax highlighting with copy buttons, quick starter chips, and conversation export (Markdown/JSON).
- 💻 **Terminal CLI Interface**: Interactive CLI (`cli.py`) with rich commands (`/context`, `/persona`, `/strategy`, `/clear`, `/exit`).

---

## 📁 Repository Structure

```
contextai-llm-chatbot/
│
├── app.py                 # Flask server & REST API endpoints (/chat, /api/config, /health)
├── llm_client.py          # Unified OpenAI & Google Gemini provider client with error mapper
├── context_manager.py     # Sliding window, running summarization, and session memory
├── cli.py                 # Interactive terminal CLI with context inspector
├── test_context.py        # Automated test suite (11 unit & integration tests)
├── requirements.txt       # Version-pinned Python dependencies
├── .env.example           # Environment template with setup guidance
├── .gitignore             # Strict exclusion of .env and local credentials
│
├── templates/
│   └── index.html         # Modern, accessible semantic single-page layout
│
└── static/
    ├── style.css          # Curated dark/light theme, glassmorphic styling, micro-animations
    └── app.js             # Client logic: Markdown parser, telemetry sync, drawer inspector
```

---

## ⚙️ Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- An API key from [Google AI Studio](https://aistudio.google.com/app/apikey) (Free Tier available) or [OpenAI Platform](https://platform.openai.com/api-keys).

### 2. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/contextai-llm-chatbot.git
cd contextai-llm-chatbot

# Optional: Create a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Credentials

Copy `.env.example` to `.env` (never commit `.env` to git):

```bash
cp .env.example .env
```

Edit `.env` with your API key:

```ini
# Google Gemini (Recommended for free tier)
GEMINI_API_KEY=AIzaSy...your_gemini_key_here
GEMINI_MODEL=gemini-2.0-flash

# OR OpenAI
OPENAI_API_KEY=sk-...your_openai_key_here
OPENAI_MODEL=gpt-4o-mini

# Context Window Settings
MAX_CONTEXT_MESSAGES=10
CONTEXT_STRATEGY=sliding_window
PORT=5000
```

> **Note**: If you provide `GEMINI_API_KEY`, ContextAI auto-activates Google Gemini. If you provide `OPENAI_API_KEY`, it auto-activates OpenAI.

### 4. Run the Web Application

```bash
python app.py
```

Open your browser at: **`http://127.0.0.1:5000`**

### 5. Run the Interactive CLI (Alternative)

```bash
python cli.py
```

---

## 🧪 Context Window Verification (5-Turn Dialogue Walkthrough)

To verify that multi-turn context persists and is managed properly, run the following test in the Web UI or CLI:

| Turn | User Input | Expected Bot Context Behavior |
|:---:|:---|:---|
| **Turn 1** | *"Hi! My name is Jordan and I am a 3rd-year CS student studying distributed consensus protocols."* | Bot acknowledges your name ("Jordan") and topic. **Context: 2/10 msgs**. |
| **Turn 2** | *"What are the primary trade-offs in the Raft protocol compared to Paxos?"* | Bot explains Raft vs Paxos. Context retains Jordan & CS student persona. **Context: 4/10 msgs**. |
| **Turn 3** | *"Can you give me a simple Python code example illustrating a leader election heartbeat?"* | Bot writes Python code with timer & vote logic. **Context: 6/10 msgs**. |
| **Turn 4** | *"What was the first thing I told you about myself in our conversation?"* | **Context Proof**: Bot accurately recalls your name is **Jordan** and you are a **3rd-year CS student studying distributed consensus**! |
| **Turn 5** | *"How does what I study relate to the code example you just wrote?"* | Bot connects the heartbeat code from Turn 3 with Jordan's study of consensus protocols from Turn 1. |

Click the **"Context: X/10 msgs"** button in the header at any time to open the **Context Window Inspector** and visually inspect:
- Exactly which messages are sent in the active context buffer
- Number of truncated messages
- Estimated tokens
- Running memory summary (if Summarization strategy is selected)

---

## 🔍 Context Window Management Strategy Details

### Strategy 1: Sliding Window Truncation (Default)
- **Concept**: Given a maximum limit $N$ (e.g., 10 messages = 5 turns), only the most recent $N$ messages are included in the prompt payload sent to the LLM.
- **Benefits**: Zero latency overhead, strictly bounded token usage, optimal for immediate relevance.
- **Formula**: `messages_sent = all_messages[-N:]`

### Strategy 2: Running Summarization (Stretch Goal)
- **Concept**: When messages exceed the window limit $N$, the older overflow messages are summarized into a concise, rolling bulleted recap.
- **Injection**: The running summary is injected directly into the system prompt:
  ```
  --- RECAP OF EARLIER CONVERSATION (Context Retention) ---
  - Jordan is a 3rd-year CS student studying distributed consensus.
  - Discussed Raft vs Paxos leader election mechanisms.
  --- END OF RECAP ---
  ```
- **Benefits**: Preserves foundational facts, user preferences, and project context across indefinitely long conversations without exceeding token ceilings.

---

## 🛡️ Error Handling

ContextAI includes defensive exception mapping across both providers:

| Scenario | HTTP Code | User-Facing Notification |
|---|:---:|---|
| **Missing API Key** | `503` | *"API key is missing! Please configure GEMINI_API_KEY or OPENAI_API_KEY in .env."* |
| **Empty / Whitespace Message** | `400` | *"Please enter a valid message."* |
| **Message Over 8,000 Chars** | `400` | *"Message is too long (maximum 8,000 characters)."* |
| **Quota / Rate Limit Exceeded** | `429` | *"AI service rate limit reached. Please wait a moment before trying again."* |
| **Invalid / Revoked API Key** | `401` | *"Invalid API key. Please check your credentials in .env."* |
| **Network / DNS Drop** | `502` | *"Could not connect to the AI service. Verify your internet connection."* |
| **API Timeout (>25s)** | `504` | *"Request timed out. Please try your message again."* |

---

## 🧪 Automated Testing

ContextAI includes a dedicated test suite verifying context window mechanics, edge cases, and API routes:

```bash
python -m unittest test_context.py -v
```

**Test Coverage**:
- `test_single_turn_addition`: Verifies user & assistant message sequencing.
- `test_multi_turn_history_persistence`: Verifies model memory across multiple turns.
- `test_sliding_window_truncation`: Validates that exactly the latest $N$ messages are kept.
- `test_running_summarization_strategy`: Verifies rolling summary generation and system prompt injection.
- `test_persona_system_prompt`: Validates persona switching.
- `test_health_endpoint`: Checks `/health` status.
- `test_config_endpoint`: Validates `/api/config` schema.
- `test_empty_message_validation`: Verifies 400 Bad Request on empty inputs.
- `test_missing_api_key_error_handling`: Verifies 503 Service Unavailable when unconfigured.
- `test_successful_chat_turn`: End-to-end mocked API completion.
- `test_api_rate_limit_error_handling`: Verifies 429 status code handling.

---

## 📡 API Reference

### `POST /chat`
Send a message and receive the context-managed reply.
```json
// Request Body
{
  "message": "Explain Raft consensus simply.",
  "conversation_id": "optional-uuid",
  "persona": "mentor",
  "strategy": "sliding_window",
  "max_messages": 10
}

// Response Body
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "reply": "Raft is a consensus algorithm designed to be easy to understand...",
  "total_messages": 2,
  "context_messages_count": 1,
  "truncated_count": 0,
  "context_limit": 10,
  "strategy": "sliding_window",
  "has_summary": false,
  "running_summary": "",
  "tokens": 42,
  "provider": "gemini",
  "model": "gemini-2.0-flash"
}
```

### `GET /api/config`
Retrieves server configuration, active provider status, personas, and available strategies.

### `GET /api/conversations`
Lists all active conversation sessions and their metadata.

### `GET /api/conversations/<id>/export?format=markdown`
Exports the entire conversation history as a formatted Markdown or JSON document.

### `GET /health`
Returns system health and provider connectivity status.

---

## 🔒 Security Best Practices

- All API keys are loaded via `python-dotenv` and accessed strictly through `os.getenv()`.
- `.env`, `.env.local`, and credential files are strictly excluded via `.gitignore`.
- `.env.example` provides an untracked, safe template with dummy placeholders.
- Text content is sanitized against XSS before injection into the DOM.

---

## 📜 License
MIT License. Created for advanced LLM integration and conversational AI testing.
