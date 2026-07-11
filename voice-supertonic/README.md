# voice-supertonic — русский синтез на Supertonic-3

Движок — **[Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)**
(pip-пакет `supertonic`, репозиторий [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic)).
Лёгкий on-device TTS (~99M параметров), инференс на **ONNX Runtime CPU** — работает на
Apple Silicon без CUDA. 31 язык, включая русский (`lang="ru"`).

## Установка
```bash
cd voice-supertonic
python3 -m venv .venv && source .venv/bin/activate
pip install supertonic soundfile
```

## Веса (не в git, ~386 МБ ONNX)
Качаются автоматически при первом запуске в `~/.cache/supertonic3/`
(`vector_estimator.onnx` 245M, `vocoder.onnx` 97M, `text_encoder.onnx` 35M,
`duration_predictor.onnx` 3.5M). Обязательно:
```bash
export HF_HUB_DISABLE_XET=1        # иначе виснет на cas-bridge.xethub.hf.co
export HF_TOKEN=hf_...
```

## Запуск

### Разово (say.py)
```bash
python say.py "Привет, как дела? Сделано, всего двадцать четыре компонента."
python say.py "Текст" --voice M1 --steps 6 --speed 1.2 --play
python say.py "Текст" --style-json voice.json   # свой голос из Voice Builder
```
Пресеты: `M1..M5`, `F1..F5` (по умолчанию `F1`). Выход — 44.1kHz mono wav в `out/`.

Флаги: `--voice`, `--speed` (больше=быстрее), `--steps` (меньше=быстрее, дефолт 6),
`--play` (afplay без окна), `--out`, `--lang`, `--style-json`, `--server`.

### Сервер (для частых вызовов, порт 8126)
Standalone-режим грузит модель ~0.2с на КАЖДЫЙ вызов. Сервер держит модель и все
10 пресетов в памяти, поэтому вызов = чистая генерация без загрузки — полное
время падает почти вдвое.
```bash
nohup python server.py > out/server.log 2>&1 &   # старт в фоне
python say.py "Текст" --voice F1 --speed 1.2 --steps 6 --server
```

### Убрать даже старт Python (~0.1с на вызов)

`python say.py --server` каждый раз заново поднимает интерпретатор. Два способа
этого избежать:

**REPL — одна живая сессия, строки из stdin** (wall ≈ время генерации):
```bash
python say.py --repl --voice F1 --speed 1.2 --steps 6 --play
# дальше просто вводишь фразы, Enter — озвучить; Ctrl-D — выход
```

**curl напрямую — вообще без Python** (фиксированный `out` → сразу afplay, без
парсинга JSON):
```bash
say() {
  curl -s "http://127.0.0.1:8126/gen" --data-urlencode "text=$*" \
    -d voice=F1 -d speed=1.2 -d steps=6 -d lang=ru -d out=out/say.wav -G >/dev/null
  afplay "$(dirname "$0")/out/say.wav" 2>/dev/null || afplay out/say.wav
}
# say "Обновление: сервер перезапущен"
```

## Клонирование голоса
**В open-source SDK НЕТ.** Доступны только 10 фиксированных пресетов. Метод
`get_voice_style_from_path()` принимает не референс-wav, а готовый `.json` со
style-векторами, который отдаёт проприетарный **Supertone Voice Builder**.
Zero-shot клонирование из произвольного `lily.wav` невозможно.

## Замеры (Apple Silicon, ONNX CPU)

Фраза «Обновление: 45 компонентов библиотеки, сервер перезапущен» (audio 5.09с),
голос F1, speed 1.2. Главный рычаг скорости — `--steps` (шаги диффузии, линейно):

| steps | gen (сервер) | RTF | полное время `say.py --server` |
|---|---|---|---|
| 6 (дефолт) | 0.64 с | 0.125 | ~0.73 с |
| 4 | 0.44 с | 0.086 | ~0.53 с |
| 3 | 0.33 с | 0.065 | ~0.43 с |
| 2 | 0.23 с | 0.046 | ~0.33 с |

Standalone (с загрузкой модели 0.23с) при steps 6 — **~0.97 с** полного времени;
сервер убирает загрузку. Ниже steps ~3 звук начинает деградировать.

| прочее | значение |
|---|---|
| load модели (standalone, из кэша) | ~0.23 с |
| load модели + 10 пресетов (сервер, разово) | ~0.3 с |
| первый запуск (скачивание весов) | ~136 с |
| размер на диске | ~386 МБ |
| потоки `intra_op` | дефолт оптимален (2/4/8 не быстрее) |

Быстро: RTF ~0.05–0.13 на чистом CPU, без GPU/MPS.

## Заметки
- Инференс — ONNX CPU. MPS/Metal не используется (и не нужен для такой скорости).
- Русский поддержан нативно (в отличие от Kokoro), текст подаётся как есть,
  ударения расставлять не требуется.
- `speed=1.05` — дефолт пакета.

## Лицензия
Код — **MIT**, модель — **BigScience OpenRAIL-M**: коммерческое использование
разрешено, NC-ограничения нет (есть только use-based запреты на вредоносное
применение). Для «быстрого красивого русского на Mac» лицензионных препятствий нет.
