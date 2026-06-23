import { getClient } from "./clients.js";

// Подстроки id, которые НЕ являются текстовыми чат-моделями (для облака).
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

const TEXT_PREFIXES = ["gpt-", "o1", "o3", "o4", "chatgpt-"];

function isCloudTextModel(id) {
  const lower = id.toLowerCase();
  if (NON_TEXT_PATTERNS.some((p) => lower.includes(p))) return false;
  return TEXT_PREFIXES.some((p) => lower.startsWith(p));
}

// Список моделей провайдера. Облако фильтруется до текстовых чат-моделей,
// у LM Studio берём всё загруженное (кроме явных эмбеддингов).
export async function listModels(provider = "cloud") {
  const client = getClient(provider);
  const res = await client.models.list();
  let ids = res.data.map((m) => m.id);

  if (provider === "cloud") {
    ids = ids.filter(isCloudTextModel).sort();
    ids.sort((a, b) => {
      const score = (id) => (id.includes("mini") || id.includes("nano") ? 0 : 1);
      return score(a) - score(b) || a.localeCompare(b);
    });
  } else {
    ids = ids.filter((id) => !/embed/i.test(id)).sort();
  }

  return ids;
}
