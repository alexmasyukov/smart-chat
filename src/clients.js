import OpenAI from "openai";

// Два провайдера с OpenAI-совместимым API:
//  - cloud: OpenAI (ключ из OPENAI_TOKEN / OPENAI_API_KEY)
//  - local: LM Studio на http://127.0.0.1:1234/v1 (ключ-заглушка)
const cloudKey = process.env.OPENAI_TOKEN || process.env.OPENAI_API_KEY;
export const LMSTUDIO_URL = process.env.LMSTUDIO_URL || "http://127.0.0.1:1234/v1";

const cloudClient = cloudKey ? new OpenAI({ apiKey: cloudKey }) : null;
const localClient = new OpenAI({ baseURL: LMSTUDIO_URL, apiKey: "lm-studio" });

export const PROVIDERS = {
  cloud: { label: "Облако (OpenAI)", client: cloudClient },
  local: { label: "Локально (LM Studio)", client: localClient },
};

export function getClient(provider) {
  const p = PROVIDERS[provider];
  if (!p) throw new Error(`Неизвестный провайдер: ${provider}`);
  if (!p.client) {
    throw new Error(`Провайдер "${provider}" недоступен (нет OPENAI_TOKEN).`);
  }
  return p.client;
}
