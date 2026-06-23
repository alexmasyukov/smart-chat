import { openai } from "./openai.js";

// Подстроки в id моделей, которые НЕ являются текстовыми чат-моделями.
const NON_TEXT_PATTERNS = [
  "audio",
  "realtime",
  "tts",
  "whisper",
  "transcribe",
  "image",
  "dall-e",
  "embedding",
  "moderation",
  "search",
  "codex",
  "computer-use",
];

// Префиксы id, которые соответствуют чат-моделям, поддерживающим текст.
const TEXT_PREFIXES = ["gpt-", "o1", "o3", "o4", "chatgpt-"];

function isTextChatModel(id) {
  const lower = id.toLowerCase();
  if (NON_TEXT_PATTERNS.some((p) => lower.includes(p))) return false;
  return TEXT_PREFIXES.some((p) => lower.startsWith(p));
}

// Запрашивает список моделей через API и оставляет только текстовые чат-модели.
export async function listTextModels() {
  const res = await openai.models.list();
  const ids = res.data
    .map((m) => m.id)
    .filter(isTextChatModel)
    .sort();

  // gpt-5-mini и прочие дешёвые mini — в начало списка.
  ids.sort((a, b) => {
    const score = (id) => (id.includes("mini") || id.includes("nano") ? 0 : 1);
    return score(a) - score(b) || a.localeCompare(b);
  });

  return ids;
}
