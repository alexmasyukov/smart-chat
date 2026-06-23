#!/usr/bin/env node
// Локальный HTTP-API сервер чата для нативной оболочки (Swift-пет) и любых клиентов.
// Стриминг ответа — через SSE (text/event-stream). «Мозг» переиспользует chat.js + mcp.js.

try {
  process.loadEnvFile?.(".env");
} catch {
  // .env не обязателен
}

import http from "node:http";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { McpHub } from "./mcp.js";
import { listModels } from "./models.js";
import { SYSTEM_PROMPT, runChatTurn } from "./chat.js";
import { PROVIDERS, getClient, LMSTUDIO_URL } from "./clients.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = join(__dirname, "..", "mcp.config.json");
const PORT = Number(process.env.PORT) || 8787;
const HOST = process.env.HOST || "127.0.0.1";

const hub = new McpHub();
let busy = false;
// Без истории чата: каждый запрос отправляется отдельно (system + текущий user).

// Доступность локального провайдера (LM Studio) — быстрый probe.
async function localAvailable() {
  try {
    const r = await fetch(`${LMSTUDIO_URL}/models`, { signal: AbortSignal.timeout(1500) });
    return r.ok;
  } catch {
    return false;
  }
}

function json(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        resolve({});
      }
    });
  });
}

// --- SSE ---
function sseStart(res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
}
function sse(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data ?? {})}\n\n`);
}

// Широковещательный канал: все подписчики /api/events (пет-облако) получают
// те же события, что и инициатор чата (CLI). Так облако зеркалит ответ модели.
const listeners = new Set();
function broadcast(event, data) {
  for (const res of listeners) {
    try {
      sse(res, event, data);
    } catch {
      // мёртвое соединение почистится по 'close'
    }
  }
}
// keep-alive, чтобы долгие SSE-подключения не отваливались по таймауту
setInterval(() => {
  for (const res of listeners) {
    try {
      res.write(":ka\n\n");
    } catch {
      // игнорируем
    }
  }
}, 20000).unref();

async function handleChat(req, res) {
  const { text, model: reqModel, provider: reqProvider } = await readBody(req);
  sseStart(res);

  if (busy) {
    sse(res, "error", { message: "Сервер занят, дождитесь ответа." });
    return res.end();
  }
  if (!text || !text.trim()) {
    sse(res, "error", { message: "Пустой запрос." });
    return res.end();
  }
  if (!reqModel) {
    sse(res, "error", { message: "Не указана модель." });
    return res.end();
  }

  const provider = reqProvider || "cloud";
  let client;
  try {
    client = getClient(provider);
  } catch (err) {
    sse(res, "error", { message: err.message });
    return res.end();
  }

  busy = true;

  // Stateless: новый массив на каждый запрос — только system + текущий user.
  // (внутри хода runChatTurn временно добавляет ответы/результаты инструментов,
  //  но между сообщениями ничего не сохраняется)
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: text },
  ];

  let closed = false;
  req.on("close", () => (closed = true));

  // fan-out: и инициатору (CLI), и всем подписчикам облака
  const fan = (event, data) => {
    if (!closed) sse(res, event, data);
    broadcast(event, data);
  };

  // облако узнаёт, что начался новый ход (по нему чистит пузырь)
  broadcast("prompt", { text });

  try {
    await runChatTurn({
      client,
      provider,
      model: reqModel,
      hub,
      messages,
      onState: (value) => fan("state", { value }),
      onToken: (t) => fan("token", { text: t }),
      onTool: (info) => fan("tool", info),
    });
    fan("done", {});
  } catch (err) {
    fan("error", { message: err.message });
  } finally {
    busy = false;
    res.end();
  }
}

const server = http.createServer(async (req, res) => {
  const { method } = req;
  const url = new URL(req.url, `http://${req.headers.host}`);
  const path = url.pathname;

  if (method === "GET" && path === "/api/health") return json(res, 200, { ok: true });

  if (method === "GET" && path === "/api/providers") {
    const local = await localAvailable();
    return json(res, 200, {
      providers: [
        { id: "cloud", label: PROVIDERS.cloud.label, available: !!PROVIDERS.cloud.client },
        { id: "local", label: PROVIDERS.local.label, available: local },
      ],
    });
  }

  if (method === "GET" && path === "/api/models") {
    const provider = url.searchParams.get("provider") || "cloud";
    try {
      const list = await listModels(provider);
      return json(res, 200, { provider, list });
    } catch (err) {
      return json(res, 502, { provider, list: [], error: err.message });
    }
  }

  if (method === "GET" && path === "/api/tools")
    return json(res, 200, { list: hub.openaiTools.map((t) => t.function.name) });
  if (method === "POST" && path === "/api/chat") return handleChat(req, res);

  // Поток событий для облака-зеркала (только чтение).
  if (method === "GET" && path === "/api/events") {
    sseStart(res);
    sse(res, "hello", { tools: hub.openaiTools.length });
    listeners.add(res);
    req.on("close", () => listeners.delete(res));
    return;
  }

  json(res, 404, { error: "not found" });
});

async function boot() {
  await hub.loadFromConfig(CONFIG_PATH);

  server.listen(PORT, HOST, () => {
    // Лог в stderr — на случай если кто-то читает stdout как данные.
    console.error(`smart-chat API слушает http://${HOST}:${PORT}`);
    console.error(`  провайдеры: cloud${PROVIDERS.cloud.client ? "" : " (нет ключа)"}, local (${LMSTUDIO_URL})`);
    console.error(`  инструментов MCP: ${hub.openaiTools.length}`);
  });
}

async function shutdown() {
  server.close();
  try {
    await hub.close();
  } catch {
    // игнорируем
  }
  process.exit(0);
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);

boot().catch((err) => {
  console.error("Не удалось запустить сервер:", err);
  process.exit(1);
});
