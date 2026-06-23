#!/usr/bin/env node
// CLI-чат: интерактивный клиент локального API-сервера (src/server.js).
// Здесь ты вводишь запросы и выбираешь модель. Облако-пет зеркалит ответ
// через широковещательный канал сервера.

try {
  process.loadEnvFile?.(".env");
} catch {
  // .env не обязателен
}

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { select, input } from "@inquirer/prompts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 8787;
const BASE = `http://127.0.0.1:${PORT}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function isUp() {
  try {
    const r = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(800) });
    return r.ok;
  } catch {
    return false;
  }
}

// Поднимает сервер в фоне (detached), если он ещё не запущен. Не убивает при выходе —
// чтобы пет-облако продолжало работать после закрытия CLI.
async function ensureServer() {
  if (await isUp()) return;
  console.log("Поднимаю API-сервер…");
  const child = spawn(process.execPath, [join(__dirname, "server.js")], {
    cwd: join(__dirname, ".."),
    env: process.env,
    detached: true,
    stdio: "ignore",
  });
  child.unref();
  for (let i = 0; i < 60; i++) {
    if (await isUp()) return;
    await sleep(300);
  }
  throw new Error("Сервер не поднялся за отведённое время.");
}

async function getModels() {
  const r = await fetch(`${BASE}/api/models`);
  return r.json(); // { list, current }
}

async function pickModel(current) {
  const { list, current: serverCurrent } = await getModels();
  if (!list || list.length === 0) throw new Error("API не вернул моделей.");
  return select({
    message: "Выберите модель:",
    choices: list.map((id) => ({ name: id, value: id })),
    default: current || serverCurrent || list[0],
  });
}

// POST /api/chat и разбор SSE-потока с колбэками.
async function streamChat(text, model, { onState, onToken, onTool }) {
  const resp = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ text, model }),
  });
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of resp.body) {
    buf += decoder.decode(chunk, { stream: true });
    let sep;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (event === "message" || !data) continue;
      let payload = {};
      try {
        payload = JSON.parse(data);
      } catch {
        // мусор пропускаем
      }
      if (event === "state") onState?.(payload.value);
      else if (event === "token") onToken?.(payload.text);
      else if (event === "tool") onTool?.(payload);
      else if (event === "error") throw new Error(payload.message || "ошибка сервера");
      else if (event === "done") return;
    }
  }
}

async function main() {
  console.log("\x1b[1m=== smart-chat ===\x1b[0m");
  await ensureServer();

  const { list } = await getModels();
  console.log(`Моделей доступно: ${list.length}. Сервер: ${BASE}`);
  let model = await pickModel();
  console.log(`\nМодель: \x1b[36m${model}\x1b[0m`);
  console.log("Команды: /model — сменить модель, /clear — очистить историю, /exit — выход.\n");

  while (true) {
    let userText;
    try {
      userText = await input({ message: "Вы:" });
    } catch {
      break; // Ctrl+C
    }

    const trimmed = userText.trim();
    if (!trimmed) continue;
    if (trimmed === "/exit") break;
    if (trimmed === "/clear") {
      await fetch(`${BASE}/api/reset`, { method: "POST" });
      console.log("История очищена.\n");
      continue;
    }
    if (trimmed === "/model") {
      model = await pickModel(model);
      console.log(`Модель: \x1b[36m${model}\x1b[0m\n`);
      continue;
    }

    let wroteToken = false;
    try {
      await streamChat(trimmed, model, {
        onState: (s) => {
          if (s === "talking" && !wroteToken) process.stdout.write("\x1b[32mБот:\x1b[0m ");
        },
        onToken: (t) => {
          wroteToken = true;
          process.stdout.write(t); // токен за токеном
        },
        onTool: (info) => {
          if (info.phase === "start") {
            process.stdout.write(
              `${wroteToken ? "\n" : ""}\x1b[90m  ⚙ вызываю ${info.name}(${info.args})\x1b[0m\n`
            );
          }
        },
      });
      console.log("\n");
    } catch (err) {
      console.error(`\n\x1b[31mОшибка:\x1b[0m ${err.message}\n`);
    }
  }

  console.log("Пока! (сервер и облако продолжают работать; остановить — pnpm stop)");
}

main().catch((err) => {
  console.error("Фатальная ошибка:", err);
  process.exit(1);
});
