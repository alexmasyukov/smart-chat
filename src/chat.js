// Системный промпт: уходит с каждым отдельным сообщением (без истории чата).
export const SYSTEM_PROMPT = `Ты — ассистент с доступом к инструментам. Выбирай инструмент по смыслу запроса из доступных (их названия и описания даны в списке инструментов);
Слова «открой», «запусти», «покажи» (и их формы) — эквивалентны и означают запуск команды;
Если запрос подходит под какой-то инструмент — обязательно вызови именно его (учитывай уточнения: ADSW/адсв, Network/нетворк и т.п.);
Вызывай один инструмент за раз. Если ничего не подходит — отвечай текстом;
Всегда отвечай на русском языке;`;

// reasoning по умолчанию отключён: минимальные «раздумья» там, где параметр поддерживается.
export function reasoningParams(model) {
  const m = model.toLowerCase();
  if (m.startsWith("gpt-5")) return { reasoning_effort: "minimal" };
  if (/^o[134]/.test(m)) return { reasoning_effort: "low" };
  return {};
}

// Один ход диалога: стримит ответ и крутит цикл вызова инструментов, пока он есть.
// Мутирует messages (добавляет ответы ассистента и результаты инструментов).
// Колбэки: onState(state), onToken(text), onTool({name, args, phase, result}).
export async function runChatTurn({ client, provider, model, hub, messages, onState, onToken, onTool }) {
  while (true) {
    onState?.("thinking");

    const stream = await client.chat.completions.create({
      model,
      messages,
      tools: hub.hasTools() ? hub.openaiTools : undefined,
      stream: true,
      // Облаку — reasoning; локальным — temperature 0 для стабильной маршрутизации.
      ...(provider === "cloud" ? reasoningParams(model) : { temperature: 0 }),
    });

    let content = "";
    const toolCalls = []; // индексируется по delta.tool_calls[].index
    let talking = false;

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (!delta) continue;

      if (delta.content) {
        if (!talking) {
          talking = true;
          onState?.("talking");
        }
        onToken?.(delta.content);
        content += delta.content;
      }

      for (const t of delta.tool_calls || []) {
        const i = t.index;
        if (!toolCalls[i]) {
          toolCalls[i] = { id: "", type: "function", function: { name: "", arguments: "" } };
        }
        const slot = toolCalls[i];
        if (t.id) slot.id = t.id;
        if (t.function?.name) slot.function.name += t.function.name;
        if (t.function?.arguments) slot.function.arguments += t.function.arguments;
      }
    }

    const calls = toolCalls.filter(Boolean);
    messages.push({
      role: "assistant",
      content: content || null,
      ...(calls.length ? { tool_calls: calls } : {}),
    });

    if (calls.length) {
      for (const tc of calls) {
        let args = {};
        try {
          args = JSON.parse(tc.function.arguments || "{}");
        } catch {
          // кривой JSON — зовём с пустыми аргументами
        }
        onState?.("working");
        onTool?.({ name: tc.function.name, args: tc.function.arguments || "", phase: "start" });
        const result = await hub.callTool(tc.function.name, args);
        onTool?.({ name: tc.function.name, phase: "done", result });
        messages.push({ role: "tool", tool_call_id: tc.id, content: result });
      }
      continue; // снова к модели с результатами инструментов
    }

    onState?.("idle");
    return content;
  }
}

// ---------------------------------------------------------------------------
// Двухпроходная маршрутизация для слабых (локальных) моделей.
// Список инструментов и описания берутся ДИНАМИЧЕСКИ из MCP — новые
// инструменты подхватываются автоматически, без правок кода.
// Проход 1: классификация запроса в один из инструментов (или none).
// Проход 2: детерминированный вызов выбранного инструмента + озвучка результата.
// ---------------------------------------------------------------------------

function tokenize(s) {
  return (s.toLowerCase().match(/[a-zа-яё]{3,}/giu) || []);
}

// Индекс: слово → множество инструментов, у которых оно есть (имя + описание).
function buildTermIndex(tools) {
  const idx = new Map();
  for (const t of tools) {
    const name = t.function.name;
    const terms = new Set([
      ...tokenize(name.replace(/_/g, " ")),
      ...tokenize(t.function.description || ""),
    ]);
    for (const term of terms) {
      if (!idx.has(term)) idx.set(term, new Set());
      idx.get(term).add(name);
    }
  }
  return idx;
}

// Быстрый детерминированный путь: если в запросе есть слово, ПРИНАДЛЕЖАЩЕЕ ровно
// одному инструменту (нетворк, network, компоненты…), сразу роутим к нему.
// Неоднозначные слова (адсв есть у двух) и общие (проекты) пропускаем — их решит модель.
function keywordRoute(tools, text) {
  const idx = buildTermIndex(tools);
  const hits = new Set();
  for (const w of tokenize(text)) {
    const set = idx.get(w);
    if (set && set.size === 1) hits.add([...set][0]);
  }
  return hits.size === 1 ? [...hits][0] : null;
}

// Классификация по живому списку инструментов: строго один из enum (structured output).
async function classify(client, model, tools, text) {
  const names = tools.map((t) => t.function.name);
  const list = tools.map((t) => `- ${t.function.name}: ${t.function.description || ""}`).join("\n");
  const prompt =
    `Реши, нужен ли инструмент для запроса, и если да — выбери самый подходящий.\n` +
    `Доступные инструменты:\n${list}\n` +
    `needsTool=false для приветствия, благодарности, болтовни или если ничего не подходит.\n` +
    `Слова «открой», «запусти», «покажи» означают запуск. Учитывай уточнения: ADSW/адсв, Network/нетворк.`;
  const schema = {
    type: "json_schema",
    json_schema: {
      name: "route",
      strict: true,
      schema: {
        type: "object",
        properties: {
          needsTool: { type: "boolean" },
          tool: { type: "string", enum: names },
        },
        required: ["needsTool", "tool"],
        additionalProperties: false,
      },
    },
  };
  const messages = [
    { role: "system", content: prompt },
    { role: "user", content: text },
  ];

  try {
    const cls = await client.chat.completions.create({
      model,
      temperature: 0,
      messages,
      response_format: schema,
    });
    const parsed = JSON.parse(cls.choices[0]?.message?.content || "{}");
    if (!parsed.needsTool) return "none";
    if (names.includes(parsed.tool)) return parsed.tool;
  } catch {
    // structured output не поддержан/сломался — фолбэк ниже
  }
  // Фолбэк: свободный ответ, ищем имя инструмента в тексте.
  const cls = await client.chat.completions.create({ model, temperature: 0, messages });
  const raw = (cls.choices[0]?.message?.content || "").toLowerCase();
  return names.find((n) => raw.includes(n.toLowerCase())) || "none";
}

export async function runRoutedTurn({ client, model, hub, text, onState, onToken, onTool }) {
  onState?.("thinking");

  // --- Проход 1: сначала уникальное ключевое слово (без модели), иначе классификатор ---
  const tool =
    keywordRoute(hub.openaiTools, text) || (await classify(client, model, hub.openaiTools, text));

  // Ничего не подошло — обычный текстовый ответ.
  if (tool === "none") {
    onState?.("talking");
    const stream = await client.chat.completions.create({
      model,
      temperature: 0,
      stream: true,
      messages: [
        { role: "system", content: "Ты ассистент. Отвечай кратко и по-русски." },
        { role: "user", content: text },
      ],
    });
    for await (const chunk of stream) {
      const t = chunk.choices[0]?.delta?.content;
      if (t) onToken?.(t);
    }
    onState?.("idle");
    return;
  }

  // --- Проход 2: детерминированный вызов выбранного инструмента + озвучка ---
  onState?.("working");
  onTool?.({ name: tool, args: "{}", phase: "start" });
  const result = await hub.callTool(tool, {});
  onTool?.({ name: tool, phase: "done", result });

  onState?.("talking");
  const stream = await client.chat.completions.create({
    model,
    temperature: 0,
    stream: true,
    messages: [
      {
        role: "system",
        content:
          "Кратко по-русски сообщи пользователю результат инструмента. " +
          "Не выдумывай ничего сверх результата.",
      },
      { role: "user", content: `Запрос: ${text}\nРезультат инструмента: ${result}` },
    ],
  });
  for await (const chunk of stream) {
    const t = chunk.choices[0]?.delta?.content;
    if (t) onToken?.(t);
  }
  onState?.("idle");
}
