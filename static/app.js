/**
 * ContextAI - Frontend Application Logic
 * Implements multi-turn interaction, context telemetry, markdown rendering,
 * drawer inspector, and conversation history management.
 */

// Application State
const state = {
    conversationId: null,
    persona: "mentor",
    strategy: "sliding_window",
    maxMessages: 10,
    configured: false,
    provider: "none",
    model: "none",
    personas: {},
    latestContext: null
};

// DOM Element References
const elements = {
    chatForm: document.getElementById("chat-form"),
    messageInput: document.getElementById("message-input"),
    sendButton: document.getElementById("send-button"),
    messagesContainer: document.getElementById("messages-container"),
    welcomeCard: document.getElementById("welcome-card"),
    notificationBanner: document.getElementById("notification-banner"),
    bannerText: document.getElementById("banner-text"),
    bannerDismiss: document.getElementById("banner-dismiss"),
    chatTitle: document.getElementById("chat-title"),
    chatSubtitle: document.getElementById("chat-subtitle"),
    telemetryBtn: document.getElementById("inspect-context-btn"),
    telemetryText: document.getElementById("telemetry-text"),
    clearChatBtn: document.getElementById("clear-chat-btn"),
    newChatBtn: document.getElementById("new-chat-btn"),
    historyList: document.getElementById("history-list"),
    refreshConvsBtn: document.getElementById("refresh-convs-btn"),
    providerBadge: document.getElementById("provider-badge"),
    providerName: document.getElementById("provider-name"),
    exportMdBtn: document.getElementById("export-md-btn"),
    themeToggleBtn: document.getElementById("theme-toggle-btn"),
    themeIcon: document.getElementById("theme-icon"),
    charCounter: document.getElementById("char-counter"),
    // Sidebar & Drawer
    sidebar: document.getElementById("sidebar"),
    mobileMenuBtn: document.getElementById("mobile-menu-btn"),
    closeSidebarBtn: document.getElementById("close-sidebar-btn"),
    contextDrawer: document.getElementById("context-drawer"),
    closeDrawerBtn: document.getElementById("close-drawer-btn"),
    drawerBackdrop: document.getElementById("drawer-backdrop"),
    // Drawer Telemetry Elements
    drawerTotalMsgs: document.getElementById("drawer-total-msgs"),
    drawerContextMsgs: document.getElementById("drawer-context-msgs"),
    drawerTruncatedMsgs: document.getElementById("drawer-truncated-msgs"),
    drawerTokens: document.getElementById("drawer-tokens"),
    drawerSystemPrompt: document.getElementById("drawer-system-prompt"),
    drawerSummarySection: document.getElementById("drawer-summary-section"),
    drawerRunningSummary: document.getElementById("drawer-running-summary"),
    drawerMessagesList: document.getElementById("drawer-messages-list"),
    // Settings Controls
    personaPicker: document.getElementById("persona-picker"),
    stratSliding: document.getElementById("strat-sliding"),
    stratSummary: document.getElementById("strat-summary"),
    windowLimitSlider: document.getElementById("window-limit-slider"),
    windowLimitVal: document.getElementById("window-limit-val")
};

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
    initTheme();
    setupEventListeners();
    await fetchConfig();
    startNewConversation();
    await loadConversationList();
});

// -------------------------------------------------------------
// Configuration & Server Discovery
// -------------------------------------------------------------
async function fetchConfig() {
    try {
        const res = await fetch("/api/config");
        if (!res.ok) throw new Error("Could not load config");
        const data = await res.json();

        state.configured = data.configured;
        state.provider = data.provider;
        state.model = data.model;
        state.personas = data.personas || {};

        updateProviderBadge();
        updateDrawerSystemPrompt();

        if (!data.configured) {
            showNotification(
                "No API Key configured! Set GEMINI_API_KEY (free at Google AI Studio) or OPENAI_API_KEY in your .env file."
            );
        }
    } catch (err) {
        console.warn("Config fetch error:", err);
        state.providerName = "Offline / Standalone";
    }
}

function updateProviderBadge() {
    const dot = elements.providerBadge.querySelector(".status-dot");
    if (state.configured) {
        dot.className = "status-dot online";
        elements.providerName.textContent = `${state.provider.toUpperCase()} (${state.model})`;
    } else {
        dot.className = "status-dot";
        elements.providerName.textContent = "No API Key Set";
    }
}

// -------------------------------------------------------------
// Message Sending & Multi-Turn Context
// -------------------------------------------------------------
async function handleSendMessage(text) {
    const message = (text || elements.messageInput.value).trim();
    if (!message || elements.sendButton.disabled) return;

    hideNotification();
    removeWelcomeCard();

    // 1. Render User Message
    appendMessageRow("user", message);

    // Reset input
    elements.messageInput.value = "";
    autoResizeTextarea();
    elements.charCounter.textContent = "0 / 8000";

    // 2. Lock controls & show typing indicator
    elements.sendButton.disabled = true;
    const typingIndicator = appendTypingIndicator();

    try {
        // 3. Send to /chat endpoint
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                conversation_id: state.conversationId,
                persona: state.persona,
                strategy: state.strategy,
                max_messages: state.maxMessages
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Failed to communicate with ContextAI.");
        }

        // 4. Update Conversation State & Context Telemetry
        state.conversationId = data.conversation_id;
        state.latestContext = data;

        typingIndicator.remove();
        appendMessageRow("assistant", data.reply, data.tokens);
        updateTelemetryUI(data);
        loadConversationList();

    } catch (error) {
        typingIndicator.remove();
        showNotification(error.message || "Failed to receive response.");
    } finally {
        elements.sendButton.disabled = false;
        elements.messageInput.focus();
    }
}

// -------------------------------------------------------------
// UI Message Rendering & Markdown Formatter
// -------------------------------------------------------------
function appendMessageRow(role, text, tokens = 0) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${role}-avatar`;
    avatar.textContent = role === "user" ? "👤" : "✦";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const header = document.createElement("div");
    header.className = "message-header";
    const speaker = role === "user" ? "You" : "ContextAI";
    header.innerHTML = `<strong>${speaker}</strong><span>${formatTime(new Date())}</span>`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "user") {
        bubble.textContent = text;
    } else {
        bubble.innerHTML = renderMarkdown(text);
        attachCodeCopyButtons(bubble);
    }

    contentDiv.appendChild(header);
    contentDiv.appendChild(bubble);

    // Assistant meta row (copy button + token indicator)
    if (role === "assistant") {
        const meta = document.createElement("div");
        meta.className = "message-meta";
        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-msg-btn";
        copyBtn.innerHTML = `<span>📋 Copy</span>`;
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(text);
            copyBtn.innerHTML = `<span>✓ Copied</span>`;
            setTimeout(() => { copyBtn.innerHTML = `<span>📋 Copy</span>`; }, 2000);
        };
        meta.appendChild(copyBtn);

        if (tokens) {
            const tokenBadge = document.createElement("span");
            tokenBadge.textContent = `• ~${tokens} tokens`;
            meta.appendChild(tokenBadge);
        }
        contentDiv.appendChild(meta);
    }

    if (role === "assistant") {
        row.appendChild(avatar);
        row.appendChild(contentDiv);
    } else {
        row.appendChild(contentDiv);
        row.appendChild(avatar);
    }

    elements.messagesContainer.appendChild(row);
    scrollToBottom();
}

function appendTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant typing-row";

    const avatar = document.createElement("div");
    avatar.className = "avatar assistant-avatar";
    avatar.textContent = "✦";

    const bubble = document.createElement("div");
    bubble.className = "typing-bubble";
    bubble.innerHTML = `
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    `;

    row.appendChild(avatar);
    row.appendChild(bubble);
    elements.messagesContainer.appendChild(row);
    scrollToBottom();
    return row;
}

// -------------------------------------------------------------
// Markdown & Code Highlighter Formatter
// -------------------------------------------------------------
function renderMarkdown(rawText) {
    if (!rawText) return "";

    let html = escapeHtml(rawText);

    // Code blocks with syntax copy button
    html = html.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><button class="code-copy-btn" onclick="copyCodeSnippet(this)">Copy</button><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Headers
    html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // Bold & Italics
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Bullet points
    html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");

    // Paragraph breaks
    html = html.replace(/\n\n+/g, "</p><p>");
    html = `<p>${html}</p>`;
    html = html.replace(/<p><\/p>/g, "");
    html = html.replace(/<p>(<h[1-3]>.*?<\/h[1-3]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<pre>.*?<\/pre>)<\/p>/gs, "$1");
    html = html.replace(/<p>(<ul>.*?<\/ul>)<\/p>/gs, "$1");

    return html;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function attachCodeCopyButtons(container) {
    // Buttons are wired inline with copyCodeSnippet
}

window.copyCodeSnippet = function(button) {
    const pre = button.parentElement;
    const code = pre.querySelector("code");
    if (code) {
        navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied!";
        setTimeout(() => { button.textContent = "Copy"; }, 2000);
    }
};

// -------------------------------------------------------------
// Context Telemetry & Drawer Inspector
// -------------------------------------------------------------
function updateTelemetryUI(data) {
    const inContext = data.context_messages_count || 0;
    const limit = data.context_limit || state.maxMessages;
    const truncated = data.truncated_count || 0;

    elements.telemetryText.textContent = `Context: ${inContext}/${limit} msgs${truncated > 0 ? ` (${truncated} trunc)` : ""}`;

    // Update Drawer
    elements.drawerTotalMsgs.textContent = data.total_messages || 0;
    elements.drawerContextMsgs.textContent = inContext;
    elements.drawerTruncatedMsgs.textContent = truncated;
    elements.drawerTokens.textContent = `~${data.tokens || (inContext * 35)}`;

    if (data.has_summary && data.running_summary) {
        elements.drawerSummarySection.hidden = false;
        elements.drawerRunningSummary.textContent = data.running_summary;
    } else {
        elements.drawerSummarySection.hidden = true;
    }

    updateDrawerSystemPrompt();
}

function updateDrawerSystemPrompt() {
    const persona = state.personas[state.persona];
    if (persona) {
        elements.drawerSystemPrompt.textContent = persona.system_prompt;
        elements.chatSubtitle.textContent = `Persona: ${persona.name}`;
    }
}

// -------------------------------------------------------------
// Conversation Life Cycle & History List
// -------------------------------------------------------------
function startNewConversation() {
    state.conversationId = crypto.randomUUID();
    state.latestContext = null;

    elements.messagesContainer.replaceChildren();
    showWelcomeCard();

    elements.chatTitle.textContent = "New Conversation";
    elements.telemetryText.textContent = `Context: 0/${state.maxMessages} msgs`;
    elements.drawerTotalMsgs.textContent = "0";
    elements.drawerContextMsgs.textContent = "0";
    elements.drawerTruncatedMsgs.textContent = "0";
    elements.drawerTokens.textContent = "~0";
    elements.drawerSummarySection.hidden = true;
    elements.drawerMessagesList.innerHTML = `<p class="empty-state-text">Send a message to inspect the context window buffer.</p>`;

    updateDrawerSystemPrompt();
    elements.messageInput.focus();
}

async function loadConversationList() {
    try {
        const res = await fetch("/api/conversations");
        if (!res.ok) return;
        const data = await res.json();
        renderHistoryList(data.conversations || []);
    } catch (err) {
        console.warn("Failed to load conversations:", err);
    }
}

function renderHistoryList(conversations) {
    elements.historyList.replaceChildren();
    if (!conversations.length) {
        elements.historyList.innerHTML = `<small style="color:var(--text-muted);padding:8px 12px;">No saved chats yet.</small>`;
        return;
    }

    conversations.forEach(conv => {
        const item = document.createElement("div");
        item.className = `history-item ${conv.id === state.conversationId ? "active" : ""}`;

        const title = document.createElement("span");
        title.className = "history-title";
        title.textContent = conv.title || "Untitled Chat";
        title.title = conv.title;
        title.onclick = () => selectConversation(conv.id);

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "delete-conv-btn";
        deleteBtn.innerHTML = "✕";
        deleteBtn.title = "Delete Chat";
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteConversation(conv.id);
        };

        item.appendChild(title);
        item.appendChild(deleteBtn);
        elements.historyList.appendChild(item);
    });
}

async function selectConversation(convId) {
    try {
        const res = await fetch(`/api/conversations/${convId}`);
        if (!res.ok) throw new Error("Conversation not found");
        const data = await res.json();
        const conv = data.conversation;

        state.conversationId = conv.id;
        state.persona = conv.persona_key || "mentor";
        state.strategy = conv.strategy || "sliding_window";
        state.maxMessages = conv.max_messages || 10;

        // Sync Sidebar Controls
        syncSettingsUI();

        // Render Conversation Messages
        removeWelcomeCard();
        elements.messagesContainer.replaceChildren();
        elements.chatTitle.textContent = conv.title;

        conv.messages.forEach(m => {
            appendMessageRow(m.role, m.content, m.tokens);
        });

        const inContext = Math.min(conv.messages.length, conv.max_messages);
        const truncated = Math.max(0, conv.messages.length - conv.max_messages);
        updateTelemetryUI({
            total_messages: conv.messages.length,
            context_messages_count: inContext,
            context_limit: conv.max_messages,
            truncated_count: truncated,
            has_summary: Boolean(conv.running_summary),
            running_summary: conv.running_summary || "",
            tokens: inContext * 35
        });

        loadConversationList();
        closeSidebar();
    } catch (err) {
        showNotification(err.message || "Failed to switch conversation.");
    }
}

async function deleteConversation(convId) {
    if (!confirm("Are you sure you want to delete this conversation?")) return;
    try {
        await fetch(`/api/conversations/${convId}`, { method: "DELETE" });
        if (state.conversationId === convId) {
            startNewConversation();
        }
        loadConversationList();
    } catch (err) {
        showNotification("Failed to delete conversation.");
    }
}

async function clearCurrentChat() {
    if (!confirm("Clear all messages in this conversation? Context memory will reset.")) return;
    if (!state.conversationId) return;

    try {
        await fetch(`/api/conversations/${state.conversationId}/clear`, { method: "POST" });
        elements.messagesContainer.replaceChildren();
        showWelcomeCard();
        updateTelemetryUI({
            total_messages: 0,
            context_messages_count: 0,
            context_limit: state.maxMessages,
            truncated_count: 0,
            has_summary: false,
            tokens: 0
        });
        loadConversationList();
    } catch (err) {
        showNotification("Failed to clear chat.");
    }
}

function exportConversation() {
    if (!state.conversationId) return;
    window.open(`/api/conversations/${state.conversationId}/export?format=markdown`, "_blank");
}

// -------------------------------------------------------------
// UI Utilities & Helper Handlers
// -------------------------------------------------------------
function showWelcomeCard() {
    if (!document.getElementById("welcome-card") && elements.welcomeCard) {
        elements.messagesContainer.appendChild(elements.welcomeCard);
    }
}

function removeWelcomeCard() {
    const card = document.getElementById("welcome-card");
    if (card) card.remove();
}

function showNotification(text) {
    elements.bannerText.textContent = text;
    elements.notificationBanner.hidden = false;
}

function hideNotification() {
    elements.notificationBanner.hidden = true;
}

function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function autoResizeTextarea() {
    const input = elements.messageInput;
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

function syncSettingsUI() {
    // Persona Pills
    document.querySelectorAll(".persona-pill").forEach(pill => {
        pill.classList.toggle("active", pill.dataset.persona === state.persona);
    });

    // Strategy
    if (state.strategy === "summarization") {
        elements.stratSummary.checked = true;
    } else {
        elements.stratSliding.checked = true;
    }

    // Slider
    elements.windowLimitSlider.value = state.maxMessages;
    elements.windowLimitVal.textContent = `${state.maxMessages} msgs`;
    updateDrawerSystemPrompt();
}

// -------------------------------------------------------------
// Theme Management
// -------------------------------------------------------------
function initTheme() {
    const saved = localStorage.getItem("contextai_theme") || "dark";
    setTheme(saved);
}

function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    elements.themeIcon.textContent = theme === "dark" ? "🌙" : "☀️";
    localStorage.setItem("contextai_theme", theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    setTheme(current === "dark" ? "light" : "dark");
}

// -------------------------------------------------------------
// Event Listeners Setup
// -------------------------------------------------------------
function setupEventListeners() {
    // Form submit
    elements.chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleSendMessage();
    });

    // Keydown in Textarea (Enter to send, Shift+Enter for newline)
    elements.messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Character Counter & Auto-resize
    elements.messageInput.addEventListener("input", () => {
        autoResizeTextarea();
        const len = elements.messageInput.value.length;
        elements.charCounter.textContent = `${len} / 8000`;
    });

    // Quick-Start Prompt Chips
    document.addEventListener("click", (e) => {
        const chip = e.target.closest(".prompt-chip");
        if (chip) {
            const prompt = chip.dataset.prompt;
            elements.messageInput.value = prompt;
            handleSendMessage(prompt);
        }
    });

    // Persona Picker
    elements.personaPicker.addEventListener("click", (e) => {
        const pill = e.target.closest(".persona-pill");
        if (!pill) return;
        state.persona = pill.dataset.persona;
        syncSettingsUI();
    });

    // Context Strategy Radio Switch
    elements.stratSliding.addEventListener("change", () => {
        state.strategy = "sliding_window";
    });
    elements.stratSummary.addEventListener("change", () => {
        state.strategy = "summarization";
    });

    // Context Window Slider
    elements.windowLimitSlider.addEventListener("input", (e) => {
        state.maxMessages = parseInt(e.target.value, 10);
        elements.windowLimitVal.textContent = `${state.maxMessages} msgs`;
    });

    // Header actions
    elements.clearChatBtn.addEventListener("click", clearCurrentChat);
    elements.newChatBtn.addEventListener("click", startNewConversation);
    elements.exportMdBtn.addEventListener("click", exportConversation);
    elements.themeToggleBtn.addEventListener("click", toggleTheme);
    elements.bannerDismiss.addEventListener("click", hideNotification);
    elements.refreshConvsBtn.addEventListener("click", loadConversationList);

    // Context Inspector Drawer toggles
    elements.telemetryBtn.addEventListener("click", openDrawer);
    elements.closeDrawerBtn.addEventListener("click", closeDrawer);
    elements.drawerBackdrop.addEventListener("click", closeDrawer);

    // Mobile Sidebar
    elements.mobileMenuBtn.addEventListener("click", openSidebar);
    elements.closeSidebarBtn.addEventListener("click", closeSidebar);
}

function openDrawer() {
    elements.contextDrawer.classList.add("open");
    elements.drawerBackdrop.classList.add("active");
    elements.contextDrawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
    elements.contextDrawer.classList.remove("open");
    elements.drawerBackdrop.classList.remove("active");
    elements.contextDrawer.setAttribute("aria-hidden", "true");
}

function openSidebar() {
    elements.sidebar.classList.add("open");
}

function closeSidebar() {
    elements.sidebar.classList.remove("open");
}
