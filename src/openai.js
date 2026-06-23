import OpenAI from "openai";

const apiKey = process.env.OPENAI_TOKEN || process.env.OPENAI_API_KEY;

if (!apiKey) {
  console.error(
    "Не найден ключ OpenAI. Задайте переменную окружения OPENAI_TOKEN (или OPENAI_API_KEY)."
  );
  process.exit(1);
}

export const openai = new OpenAI({ apiKey });
