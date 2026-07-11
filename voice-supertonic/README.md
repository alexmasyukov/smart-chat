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
```bash
python say.py "Привет, как дела? Сделано, всего двадцать четыре компонента."
python say.py "Текст" --voice M1 --steps 8 --speed 1.05
python say.py "Текст" --style-json voice.json   # свой голос из Voice Builder
```
Пресеты: `M1..M5`, `F1..F5` (по умолчанию `F1`). Выход — 44.1kHz mono wav в `out/`.

## Клонирование голоса
**В open-source SDK НЕТ.** Доступны только 10 фиксированных пресетов. Метод
`get_voice_style_from_path()` принимает не референс-wav, а готовый `.json` со
style-векторами, который отдаёт проприетарный **Supertone Voice Builder**.
Zero-shot клонирование из произвольного `lily.wav` невозможно.

## Замеры (Apple Silicon, ONNX CPU, steps=8)
| метрика | значение |
|---|---|
| load (модель в кэше) | ~0.22 с |
| gen | ~0.84 с |
| audio | ~5.1 с |
| **RTF** | **~0.16** |
| первый запуск (скачивание весов) | ~136 с |
| размер на диске | ~386 МБ |

Быстро: RTF ~0.16 на чистом CPU, без GPU/MPS. Главный рычаг скорости —
`total_steps` (шаги диффузии, дефолт 8).

## Заметки
- Инференс — ONNX CPU. MPS/Metal не используется (и не нужен для такой скорости).
- Русский поддержан нативно (в отличие от Kokoro), текст подаётся как есть,
  ударения расставлять не требуется.
- `speed=1.05` — дефолт пакета.

## Лицензия
Код — **MIT**, модель — **BigScience OpenRAIL-M**: коммерческое использование
разрешено, NC-ограничения нет (есть только use-based запреты на вредоносное
применение). Для «быстрого красивого русского на Mac» лицензионных препятствий нет.
