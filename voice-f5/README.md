# voice-f5 — русский синтез/клонирование на F5-TTS

Движок — официальный **[SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)** (пакет `f5-tts`),
веса — русский файнтюн **[Misha24-10/F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN)**
(чекпоинт `F5TTS_v1_Base_v4_winter/model_212000`). Работает на Apple Silicon.

## Установка
```bash
cd voice-f5
python3 -m venv .venv && source .venv/bin/activate
pip install "setuptools<81" f5-tts    # setuptools<81 нужен для perth/pkg_resources
```

## Веса (не в git, ~1.3 ГБ)
```bash
export HF_TOKEN=...
BASE=https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  "$BASE/F5TTS_v1_Base_v4_winter/model_212000.safetensors" -o ckpt/model_212000.safetensors
# vocab.txt уже лежит в ckpt/ (в репозитории)
```

## Референс
`ref/lily.wav` (9с), `ref/lily_5s.wav`, `ref/lily_3s.wav` — один голос разной длины.
F5 оценивает темп речи по референсу: короче референс → быстрее речь и быстрее синтез.

## Запуск

**Сервер** (модель в памяти, кеш транскрипции референса, рекомендуется):
```bash
python server.py                       # http://127.0.0.1:8124
python say.py "Текст голосом Lily"     # клиент
```

**Разово** (грузит модель на запуск):
```bash
python generate.py "Текст" --device mps
```

## Скорость (mps, замерено bench.py / bench2.py)
- Главный рычаг — `nfe_step` (шаги диффузии), линейный: 32→18.8с, **16→9.2с**, 8→4.65с.
- Короткий референс: ref9→ref3 даёт RTF 0.87→0.58.
- Дефолт сервера: **nfe=16 + референс 3с** → ~5с синтеза на ~8с речи (RTF ~0.58).
- fp16 на mps и cfg=1 ускорения **не** дают (проверено).

## Заметки
- На `mps` бывает мусорный звук у ванильного F5 из-за pinyin-словаря — здесь берётся
  кастомный русский `vocab.txt` (`token: custom`), звук чистый.
- Ударения: `+` перед гласной (`молок+о`). Для авторасстановки — RUAccent.
