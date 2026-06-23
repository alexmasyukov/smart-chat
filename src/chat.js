import { openai } from "./openai.js";

// Системный промпт: модель работает диспетчером инструментов ALERT-APP
// и понимает запрос по смыслу, а не по точным словам.
export const SYSTEM_PROMPT = `Ты — диспетчер вызова инструментов MCP-приложения «ALERT-APP».

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
export async function runChatTurn({ model, hub, messages, onState, onToken, onTool }) {
  while (true) {
    onState?.("thinking");

    const stream = await openai.chat.completions.create({
      model,
      messages,
      tools: hub.hasTools() ? hub.openaiTools : undefined,
      stream: true,
      ...reasoningParams(model),
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
