const modelSearch = document.querySelector("#model-search");
const modelSelect = document.querySelector("#model-select");
const modelDetails = document.querySelector("#model-details");
const messagesEl = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const clearChatButton = document.querySelector("#clear-chat");
const statusEl = document.querySelector("#status");

let models = [];
let messages = [];
let socket = null;
let awaitingResponse = false;

function setStatus(text) {
  statusEl.textContent = text;
}

function selectedModel() {
  return models.find((model) => model.key === modelSelect.value);
}

function renderModels() {
  const query = modelSearch.value.trim().toLowerCase();
  const filtered = models.filter((model) => {
    return `${model.key} ${model.label} ${model.company}`
      .toLowerCase()
      .includes(query);
  });

  modelSelect.replaceChildren();
  for (const model of filtered) {
    const option = document.createElement("option");
    option.value = model.key;
    option.textContent = `${model.key} — ${model.label}`;
    modelSelect.append(option);
  }

  if (!filtered.some((model) => model.key === modelSelect.value)) {
    modelSelect.value = filtered[0]?.key ?? "";
  }

  renderModelDetails();
}

function renderModelDetails() {
  const model = selectedModel();
  if (!model) {
    modelDetails.textContent = "No model selected.";
    return;
  }

  const badges = [
    model.reasoning ? "reasoning" : null,
    model.supports_temperature ? "temperature" : null,
    model.supports_tools ? "tools" : null,
    model.supports_images ? "images" : null,
    model.supports_files ? "files" : null,
    model.internal_only ? "internal" : "public",
    model.open_source ? "open source" : "closed source",
  ].filter(Boolean);

  modelDetails.innerHTML = `
    <div class="detail-title">${model.label}</div>
    <div>${model.company} · ${model.provider}</div>
    <div>${model.context_window.toLocaleString()} context · ${model.max_tokens.toLocaleString()} max output</div>
    <div class="badges">
      ${badges.map((badge) => `<span class="badge">${badge}</span>`).join("")}
    </div>
  `;
}

function metadataText(metadata) {
  const parts = [];
  if (metadata.duration_seconds !== null && metadata.duration_seconds !== undefined) {
    parts.push(`${metadata.duration_seconds}s`);
  }
  if (metadata.total_input_tokens || metadata.total_output_tokens) {
    parts.push(
      `${metadata.total_input_tokens ?? 0} input tokens · ${metadata.total_output_tokens ?? 0} output tokens`,
    );
  }
  if (metadata.cost?.total !== null && metadata.cost?.total !== undefined) {
    parts.push(`$${metadata.cost.total.toFixed(6)}`);
  }
  return parts.join(" · ");
}

function renderMessages() {
  messagesEl.replaceChildren();

  if (messages.length === 0 && !awaitingResponse) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Choose a model and send the first message.";
    messagesEl.append(empty);
    return;
  }

  for (const message of messages) {
    const bubble = document.createElement("div");
    bubble.className = `message ${message.role}`;
    bubble.textContent = message.content;

    if (message.reasoning) {
      const reasoning = document.createElement("div");
      reasoning.className = "reasoning";
      reasoning.textContent = message.reasoning;
      bubble.append(reasoning);
    }

    if (message.metadata) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = metadataText(message.metadata);
      bubble.append(meta);
    }

    messagesEl.append(bubble);
  }

  if (awaitingResponse) {
    const bubble = document.createElement("div");
    bubble.className = "message assistant";
    bubble.textContent = "Thinking...";
    messagesEl.append(bubble);
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setAwaiting(value) {
  awaitingResponse = value;
  sendButton.disabled = value || !modelSelect.value;
  messageInput.disabled = value;
  renderMessages();
}

function openSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${window.location.host}/ws/chat`);

  socket.addEventListener("open", () => setStatus("Connected"));
  socket.addEventListener("close", () => setStatus("Disconnected"));
  socket.addEventListener("error", () => setStatus("Connection error"));
  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    setAwaiting(false);

    if (data.type === "error") {
      setStatus(typeof data.error === "string" ? data.error : "Request failed");
      return;
    }

    messages.push({
      role: "assistant",
      content: data.output_text || "",
      reasoning: data.reasoning,
      metadata: data.metadata,
    });
    setStatus("Ready");
    renderMessages();
  });
}

async function loadModels() {
  const response = await fetch("/api/models");
  models = await response.json();
  renderModels();
  setStatus("Ready");
  sendButton.disabled = !modelSelect.value;
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = messageInput.value.trim();
  if (!content || !socket || socket.readyState !== WebSocket.OPEN || !modelSelect.value) {
    return;
  }

  const payloadMessages = [...messages, { role: "user", content }].map((message) => ({
    role: message.role,
    content: message.content,
  }));

  messages.push({ role: "user", content });
  messageInput.value = "";
  setStatus("Waiting for model...");
  setAwaiting(true);
  socket.send(JSON.stringify({ model: modelSelect.value, messages: payloadMessages }));
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    chatForm.requestSubmit();
  }
});

modelSearch.addEventListener("input", renderModels);
modelSelect.addEventListener("change", renderModelDetails);
clearChatButton.addEventListener("click", () => {
  messages = [];
  setStatus("Ready");
  renderMessages();
});

openSocket();
loadModels().catch((error) => {
  setStatus(error instanceof Error ? error.message : "Could not load models");
});
renderMessages();
