"""
ContextAI - Terminal CLI Interface.
Interactive command-line interface for testing multi-turn conversation
and context window management directly from the terminal.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

from llm_client import llm_service, LLMClientError
from context_manager import (
    conversation_manager,
    PERSONAS,
    DEFAULT_PERSONA,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_STRATEGY
)


def print_banner():
    print("=" * 65)
    print("  ✦ ContextAI - Multi-turn LLM Chatbot CLI ✦")
    print("=" * 65)
    info = llm_service.get_info()
    if info["configured"]:
        print(f"  Provider: {info['provider'].upper()} | Model: {info['model']}")
    else:
        print("  [!] Warning: No API key detected in .env!")
        print("      Set GEMINI_API_KEY or OPENAI_API_KEY in your .env file.")
    print("  Commands:")
    print("    /help      - Show this help menu")
    print("    /context   - Inspect active context window & memory stats")
    print("    /strategy  - Toggle between 'sliding_window' and 'summarization'")
    print("    /persona   - Switch assistant persona")
    print("    /clear     - Reset the current conversation history")
    print("    /exit      - Quit the CLI")
    print("=" * 65 + "\n")


def main():
    print_banner()

    conv = conversation_manager.get_or_create(
        conversation_id=None,
        persona_key=DEFAULT_PERSONA,
        strategy=DEFAULT_STRATEGY,
        max_messages=DEFAULT_MAX_MESSAGES
    )
    cid = conv["id"]

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ContextAI. Goodbye!")
            break

        if not user_input:
            continue

        # Handle Slash Commands
        cmd = user_input.lower()
        if cmd == "/exit":
            print("Goodbye!")
            break

        elif cmd == "/help":
            print_banner()
            continue

        elif cmd == "/context":
            ctx = conversation_manager.prepare_context(cid)
            print("\n--- [Context Window Status] ---")
            print(f"  Conversation ID : {cid}")
            print(f"  Strategy        : {ctx['strategy']}")
            print(f"  Total Messages  : {ctx['total_messages']}")
            print(f"  Messages in Context: {ctx['context_messages_count']} / {conv['max_messages']}")
            print(f"  Truncated Messages : {ctx['truncated_count']}")
            print(f"  Has Summary     : {ctx['has_summary']}")
            if ctx.get("running_summary"):
                print(f"  Running Summary : {ctx['running_summary']}")
            print("-------------------------------\n")
            continue

        elif cmd == "/clear":
            conversation_manager.clear_conversation(cid)
            print(" Conversation history cleared! Fresh context started.\n")
            continue

        elif cmd == "/strategy":
            new_strat = "summarization" if conv["strategy"] == "sliding_window" else "sliding_window"
            conversation_manager.update_settings(cid, strategy=new_strat)
            print(f" Switched strategy to: {new_strat}\n")
            continue

        elif cmd == "/persona":
            print("\nAvailable Personas:")
            for k, p in PERSONAS.items():
                active = " (Active)" if k == conv["persona_key"] else ""
                print(f"  - {k}: {p['name']} -- {p['description']}{active}")
            choice = input("Enter persona name > ").strip().lower()
            if choice in PERSONAS:
                conversation_manager.update_settings(cid, persona_key=choice)
                print(f" Persona switched to '{PERSONAS[choice]['name']}'\n")
            else:
                print(" Invalid persona. Keeping current.\n")
            continue

        # Normal Chat Message
        conversation_manager.add_user_message(cid, user_input)
        context_data = conversation_manager.prepare_context(cid, llm_service=llm_service)

        print("\nContextAI is thinking...", end="\r", flush=True)

        try:
            reply, tokens = llm_service.generate_chat_response(
                messages=context_data["context_messages"],
                system_prompt=context_data["effective_system_prompt"],
                temperature=0.7
            )
            # Clear the thinking indicator
            print(" " * 30, end="\r")

            conversation_manager.add_assistant_message(cid, reply, tokens=tokens)

            # Display response
            print(f"ContextAI > {reply}\n")
            print(f"[Context: {context_data['context_messages_count']}/{conv['max_messages']} msgs | Truncated: {context_data['truncated_count']} | Tokens: ~{tokens}]\n")

        except LLMClientError as e:
            print(" " * 30, end="\r")
            print(f"\n[!] Error ({e.status_code}): {e.message}\n")
        except Exception as e:
            print(" " * 30, end="\r")
            print(f"\n[!] Unexpected Error: {e}\n")


if __name__ == "__main__":
    main()
