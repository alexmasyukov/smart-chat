#!/usr/bin/env node
// Загружаем .env (если есть) до импорта модулей, читающих ключ.
try {
  process.loadEnvFile?.(".env");
} catch {
  // .env не обязателен — ключ может быть в окружении
}

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { select, input } from "@inquirer/prompts";
import { listTextModels } from "./models.js";
import { McpHub } from "./mcp.js";
import { openai } from "./openai.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = join(__dirname, "..", "mcp.config.json");

const SYSTEM_PROMPT = `Ты — диспетчер вызова инструментов MCP-приложения «ALERT-APP».

Твоя задача — понять по смыслу, что хочет пользователь, и вызвать подходящий инструмент.
Ты НЕ ведёшь беседу и НЕ пересказываешь содержимое — ты определяешь намерение и вызываешь инструмент.

Понимай запрос по-человечески, а не по точным словам. Пользователь может формулировать
свободно ("а покажи все проекты, какие есть", "что там по компонентам?", "глянь раг"),
с синонимами, опечатками, в любом падеже и порядке слов. Триггерься на СМЫСЛ.

Доступные инструменты и когда их звать:
- show_projects — когда речь про проекты (проект, проекты, проектики, "что за проекты", список проектов и т.п.).
- show_components — когда речь про компоненты (компонент, компоненты, детали, составные части, "из чего состоит" и т.п.).
- show_rag — когда упомянут rag / RAG или любая русская словоформа с корнем "раг"/"рэг"/"рак"
  в ЛЮБОМ падеже (раг, рага, рагу, рагом, раге, рэг, рак, рака — всё это про RAG),
  а также формулировки про базу знаний / поиск по документам.

Правила:
- Если намерение ясно — сразу вызывай нужный инструмент, без лишних слов.
- Если в запросе несколько тем (например, и проекты, и компоненты) — вызови все подходящие инструменты.
- Если запрос вообще не относится ни к одному инструменту — коротко уточни, что нужно.
- Не выдумывай данные: показывай только то, что вернул инструмент.`;

async function pickModel(current) {
  console.log("Запрашиваю список моделей через API…");
  const models = await listTextModels();
  if (models.length === 0) {
    throw new Error("API не вернул ни одной текстовой модели.");
  }
  return select({
    message: "Выберите модель:",
    choices: models.map((id) => ({ name: id, value: id })),
    default: current && models.includes(current) ? current : models[0],
  });
}

// Reasoning по умолчанию отключён: минимальные «раздумья» для моделей,
// которые это поддерживают (gpt-5*, o-серия). Для обычных моделей — ничего.
function reasoningParams(model) {
  const m = model.toLowerCase();
  if (m.startsWith("gpt-5")) return { reasoning_effort: "minimal" };
  if (/^o[134]/.test(m)) return { reasoning_effort: "low" };
  return {};
}

// Читает стрим OpenAI: печатает текст по токенам и накапливает tool_calls.
// Возвращает { content, toolCalls } для восстановления сообщения ассистента.
async function streamResponse(model, hub, messages) {
  const stream = await openai.chat.completions.create({
    model,
    messages,
    tools: hub.hasTools() ? hub.openaiTools : undefined,
    stream: true,
    ...reasoningParams(model),
  });

  let content = "";
  const toolCalls = []; // индексируется по delta.tool_calls[].index
  let printedLabel = false;

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta;
    if (!delta) continue;

    if (delta.content) {
      if (!printedLabel) {
        process.stdout.write("\x1b[32mБот:\x1b[0m ");
        printedLabel = true;
      }
      process.stdout.write(delta.content); // токен за токеном, как в ChatGPT
      content += delta.content;
    }

    for (const tcDelta of delta.tool_calls || []) {
      const i = tcDelta.index;
      if (!toolCalls[i]) {
        toolCalls[i] = { id: "", type: "function", function: { name: "", arguments: "" } };
      }
      const slot = toolCalls[i];
      if (tcDelta.id) slot.id = tcDelta.id;
      if (tcDelta.function?.name) slot.function.name += tcDelta.function.name;
      if (tcDelta.function?.arguments) slot.function.arguments += tcDelta.function.arguments;
    }
  }

  if (printedLabel) process.stdout.write("\n");
  return { content, toolCalls: toolCalls.filter(Boolean) };
}

// Один ход диалога: гоняем модель в цикле, пока она вызывает инструменты.
async function runTurn(model, hub, messages) {
  while (true) {
    const { content, toolCalls } = await streamResponse(model, hub, messages);

    messages.push({
      role: "assistant",
      content: content || null,
      ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
    });

    if (toolCalls.length) {
      for (const tc of toolCalls) {
        let args = {};
        try {
          args = JSON.parse(tc.function.arguments || "{}");
        } catch {
          // оставляем пустые аргументы при кривом JSON
        }
        console.log(`\x1b[90m  ⚙ вызываю ${tc.function.name}(${tc.function.arguments || ""})\x1b[0m`);
        const result = await hub.callTool(tc.function.name, args);
        messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: result,
        });
      }
      continue; // снова к модели с результатами инструментов
    }

    return; // текст уже напечатан в стриме
  }
}

async function main() {
  console.log("\x1b[1m=== smart-chat ===\x1b[0m");

  const hub = new McpHub();
  console.log("Подключаю MCP-серверы…");
  await hub.loadFromConfig(CONFIG_PATH);
  if (!hub.hasTools()) {
    console.log("  (инструментов нет — чат работает в обычном режиме)");
  }

  let model = await pickModel();
  console.log(`\nМодель: \x1b[36m${model}\x1b[0m`);
  console.log("Команды: /model — сменить модель, /clear — очистить историю, /exit — выход.\n");

  const messages = [{ role: "system", content: SYSTEM_PROMPT }];

  const cleanup = async () => {
    await hub.close();
    process.exit(0);
  };
  process.on("SIGINT", cleanup);

  while (true) {
    let userText;
    try {
      userText = await input({ message: "Вы:" });
    } catch {
      break; // Ctrl+C в промпте
    }

    const trimmed = userText.trim();
    if (!trimmed) continue;

    if (trimmed === "/exit") break;
    if (trimmed === "/clear") {
      messages.length = 1; // оставляем system
      console.log("История очищена.\n");
      continue;
    }
    if (trimmed === "/model") {
      model = await pickModel(model);
      console.log(`Модель: \x1b[36m${model}\x1b[0m\n`);
      continue;
    }

    messages.push({ role: "user", content: trimmed });

    try {
      await runTurn(model, hub, messages);
      console.log(); // пустая строка после ответа
    } catch (err) {
      console.error(`\x1b[31mОшибка:\x1b[0m ${err.message}\n`);
      // откатываем неудачное сообщение, чтобы не ломать историю
      messages.pop();
    }
  }

  await hub.close();
  console.log("Пока!");
}

main().catch((err) => {
  console.error("Фатальная ошибка:", err);
  process.exit(1);
});
