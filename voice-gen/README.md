# voice-gen — генерация/клонирование голоса

Zero-shot клонирование голоса на модели [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
(600+ языков, диффузионная LM на базе Qwen3-0.6B). Работает локально на Apple Silicon (mps).

## Установка

```bash
cd voice-gen
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchaudio==2.8.0 omnivoice
```

Веса (~3.2 ГБ) качаются автоматически при первом запуске в кэш HuggingFace.
Если скачивание виснет — `export HF_HUB_DISABLE_XET=1`.

## Референс

`ref/lily.wav` — голос Lily (ElevenLabs), обрезан до 9с, 24 кГц, моно.
Исходник конвертировался так:

```bash
ffmpeg -i source.mp3 -t 9 -ar 24000 -ac 1 ref/lily.wav
```

## Запуск

```bash
source .venv/bin/activate
python generate.py "Текст для озвучки голосом Lily"
```

Результат — в `out/clone_<timestamp>.wav`. `ref_text` не передаём:
модель сама транскрибирует референс через Whisper.

### Режимы OmniVoice

- **Voice cloning** — `ref_audio` + (опц.) `ref_text`
- **Voice design** — без референса, через `instruct="female, low pitch, british accent"`
- **Auto voice** — только `text`
