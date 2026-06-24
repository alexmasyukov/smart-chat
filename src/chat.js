// Системный промпт: уходит с каждым отдельным сообщением (без истории чата).
export const SYSTEM_PROMPT = `Ты — ассистент с доступом к инструментам. Выбирай инструмент по смыслу запроса;
- show_components — раздел «Компоненты»;
- show_projects — раздел «Проекты»;
- show_rag — раздел «RAG-система», "рак", "рак система";
- open_adsw — открыть папку ADSW (адсв);
Слова «открой», «запусти», «покажи» (и их формы) — эквивалентны и означают запуск команды;
Если в запросе есть такое слово вместе с названием раздела/папки — обязательно вызови нужный инструмент;
Примеры;
- «покажи компоненты» / «открой компоненты» / «запусти компоненты» → show_components;
- «покажи проекты» / «открой проекты» / «запусти проекты» → show_projects;
- «покажи раг» / «открой рак» / «запусти rag» → show_rag;
- «открой адсв» / «запусти adsw» / «покажи папку ADSW» → open_adsw;
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
// Проход 1: классификация запроса в один ярлык (без инструментов — модели проще).
// Проход 2: по ярлыку детерминированно вызываем нужный инструмент кодом и
//           просим модель кратко озвучить результат по-русски.
// ---------------------------------------------------------------------------

const SECTIONS = [
  { label: "components", tool: "show_components", match: ["components", "компонент"] },
  { label: "projects", tool: "show_projects", match: ["projects", "проект"] },
  { label: "rag", tool: "show_rag", match: ["rag", "раг", "рак"] },
  { label: "adsw", tool: "open_adsw", match: ["adsw", "адсв"] },
];

const CLASSIFY_PROMPT = `Ты — классификатор. По сообщению пользователя выбери ОДИН раздел:
- components — речь о компонентах;
- projects — речь о проектах;
- rag — речь о RAG-системе (раг, рак, база знаний);
- adsw — открыть папку ADSW (адсв);
- none — обычный разговор/приветствие или ничего из перечисленного.
Слова «открой», «запусти», «покажи» означают запуск и не меняют выбор раздела.
Верни только поле section.`;

const ROUTE_SCHEMA = {
  type: "json_schema",
  json_schema: {
    name: "route",
    strict: true,
    schema: {
      type: "object",
      properties: { section: { type: "string", enum: ["components", "projects", "rag", "adsw", "none"] } },
      required: ["section"],
      additionalProperties: false,
    },
  },
};

// Находит реальное имя функции инструмента в хабе (с учётом возможного префикса).
function resolveToolName(hub, bare) {
  const t = hub.openaiTools.find(
    (x) => x.function.name === bare || x.function.name.endsWith(`__${bare}`)
  );
  return t?.function.name || bare;
}

// Быстрый детерминированный проход: явное слово раздела в начале слова.
// (граница слова, чтобы «рак» не ловился внутри «практика» и т.п.)
function keywordSection(text) {
  const t = text.toLowerCase();
  for (const s of SECTIONS) {
    for (const term of s.match) {
      if (new RegExp(`(^|[^a-zа-яё])${term}`, "i").test(t)) return s.label;
    }
  }
  return null;
}

// Классификация: сначала по ключевым словам, затем модель (structured output), затем фолбэк.
async function classify(client, model, text) {
  const kw = keywordSection(text);
  if (kw) return kw;

  try {
    const cls = await client.chat.completions.create({
      model,
      temperature: 0,
      messages: [
        { role: "system", content: CLASSIFY_PROMPT },
        { role: "user", content: text },
      ],
      response_format: ROUTE_SCHEMA,
    });
    const parsed = JSON.parse(cls.choices[0]?.message?.content || "{}");
    if (SECTIONS.some((s) => s.label === parsed.section) || parsed.section === "none") {
      return parsed.section;
    }
  } catch {
    // структурированный вывод не поддержан/сломался — фолбэк ниже
  }
  // Фолбэк: свободный ответ + сопоставление по подстроке.
  const cls = await client.chat.completions.create({
    model,
    temperature: 0,
    messages: [
      { role: "system", content: CLASSIFY_PROMPT },
      { role: "user", content: text },
    ],
  });
  const raw = (cls.choices[0]?.message?.content || "").toLowerCase();
  return SECTIONS.find((s) => s.match.some((m) => raw.includes(m)))?.label || "none";
}

export async function runRoutedTurn({ client, model, hub, text, onState, onToken, onTool }) {
  onState?.("thinking");

  // --- Проход 1: классификация (строго один раздел) ---
  const section = await classify(client, model, text);
  const hit = SECTIONS.find((s) => s.label === section);

  // Ничего не подошло — обычный текстовый ответ.
  if (!hit) {
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

  // --- Проход 2: детерминированный вызов инструмента + озвучка результата ---
  onState?.("working");
  const fnName = resolveToolName(hub, hit.tool);
  onTool?.({ name: fnName, args: "{}", phase: "start" });
  const result = await hub.callTool(fnName, {});
  onTool?.({ name: fnName, phase: "done", result });

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
