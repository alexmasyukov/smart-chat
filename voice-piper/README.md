# voice-piper — быстрый русский TTS на Piper

[Piper](https://github.com/OWASP/piper) (VITS, ONNX) с голосом **`ru_RU-irina-medium`**.
CPU, без сервера, ~40× быстрее реального времени. Не клонирует голос (фиксированный
пресет), зато стабилен на коротких фразах (жёсткое выравнивание, не диффузия).

## Установка
```bash
cd voice-piper
python3 -m venv .venv && source .venv/bin/activate
pip install piper-tts
python -m piper.download_voices ru_RU-irina-medium --data-dir voices
```

Голос (~60 МБ `.onnx` + json) кладётся в `voices/`, в git не коммитится.

## Запуск
```bash
source .venv/bin/activate
python say.py "Сделано, всего 24 компонента"
```

Результат — `out/piper_<ts>.wav` (по умолчанию НЕ проигрывается).

## Параметры
- `--speed 1..10` — скорость (**дефолт 7**). 1 = очень быстро (0.1), 10 = норма (1.0).
- `--length FLOAT` — точный `length_scale` (переопределяет `--speed`).
- `--out PATH` — путь файла
- `--play` — проиграть сразу через `afplay` (без окна плеера)
- `--noise`, `--noise-w` — вариативность интонации

```bash
python say.py "Текст" --speed 3          # быстро
python say.py "Текст" --speed 10 --play  # нормальный темп + сразу проиграть
```

## Замеры (CPU, M-series)
- Загрузка модели: ~0.4с
- Синтез: RTF ~0.02 (10с речи за ~0.2с)
- Сервер не нужен — модель грузится мгновенно на каждый вызов.

## Заметки
- Голос **irina** (RHVoice-датасет), автоматические ударения. Другие русские голоса:
  `ru_RU-dmitri-medium`, `ru_RU-ruslan-medium`, `ru_RU-denis-medium`.
- Иногда путает ударение в омографах — редко, лечится ручной разметкой.
