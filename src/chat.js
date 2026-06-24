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
