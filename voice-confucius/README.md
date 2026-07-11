# voice-confucius — Confucius4-TTS (MLX, Apple Silicon)

Многоязычный zero-shot TTS с клонированием голоса
[netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS),
MLX-порт для Apple Silicon через пакет [mlx-audio](https://github.com/Blaizzy/mlx-audio).

- Веса: [mlx-community/Confucius4-TTS-mlx-int8](https://huggingface.co/mlx-community/Confucius4-TTS-mlx-int8) (int8, ~2.6 ГБ)
- 14 языков: zh, en, ja, ko, de, fr, es, id, it, th, pt, **ru**, ms, vi
- Клонирование голоса из референс-wav, транскрипт референса НЕ нужен (unconstrained)
- Выход 22050 Гц, лицензия **Apache-2.0** (коммерция разрешена)

## Оценка качества (русский, клон Lily)

- ✅ **Идеальное произношение и ударения**, естественная тональность/интонация —
  лучшее, что видели среди протестированных по расстановке ударений.
- ❌ **Звук сжатый, «как из трубы»** — глухой, телефонного качества. Похоже на
  артефакт int8-кванта + вокодера на 22050 Гц. Разборчиво и выразительно, но
  «жирного» чистого звука нет.

Вывод: отличная просодия/ударения, но тембр испорчен компрессией — для
продакшена по звуку уступает, для черновиков/озвучки текста годится.

## Скорость и память (сервер, тёплый)

| Тип фразы | Символов | Аудио | Генерация | RTF |
|---|---|---|---|---|
| очень короткий | 7 | 1.00с | 1.68с | 1.68 |
| короткий | 17 | 1.59с | 2.10с | 1.32 |
| средний | 42 | 2.81с | 3.44с | 1.23 |
| длинный | 95 | 5.19с | 6.01с | 1.16 |

RTF ~1.1–1.3 (чуть медленнее реального времени), длинные тексты эффективнее.
Пик памяти во время генерации ~5.8 ГБ, в простое сервер откатывается к ~2.9 ГБ.

## Установка

```bash
cd voice-confucius
python3 -m venv .venv && source .venv/bin/activate
pip install mlx-audio soundfile
```

Confucius4 поддерживается начиная с mlx-audio v0.4.5.

## Веса

Качаются автоматически в кэш HuggingFace при первом запуске (~2.6 ГБ, медленно).
Из-за Xet-бага нужны переменные окружения:

```bash
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=hf_...
```

## Референс

`ref/lily_3s.wav` (3с) — дефолт. Референс **ре-энкодится на каждый вызов**
(w2v-bert по ref-аудио), поэтому короткий 3с почти вдвое дешевле 9-секундного
`ref/lily.wav` (RTF 1.25 против 3.45). Модель сама ресемплит до 16 кГц.

## Запуск

### Разово (say.py)

```bash
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1 HF_TOKEN=hf_...
python say.py "Привет, как дела?"                 # дефолтный голос (Lily 3с)
python say.py "Текст" --ref ref/lily.wav --play   # свой референс + проиграть
```

Результат — в `out/confucius_<lang>_<timestamp>.wav`. Замеры load/gen/RTF в stdout.

### Сервер (рекомендуется)

Модель грузится один раз и **прогревается** при старте (первая генерация
компилирует Metal-кернелы — холодный RTF ~7, после прогрева ~1.2). Признаки
референса кешируются.

```bash
# старт в фоне
export HF_HUB_DISABLE_XET=1 HF_TOKEN=hf_...
nohup python server.py > out/server.log 2>&1 &   # порт 8125

# генерация через тёплый сервер
python say.py "Текст" --server --play
```

Флаги `say.py`: `--server`, `--play` (afplay), `--ref`, `--lang`, `--out`,
`--temperature`, `--top-k`, `--top-p`, `--rep-pen`, `--seed`.

## Память (важно)

MLX по умолчанию кеширует освобождённые Metal-буферы **без лимита** — за
несколько генераций unified-память раздувается до **десятков ГБ** (в `ps` не
видно, только в Activity Monitor / `top`). `server.py` это лечит:

- `mx.set_cache_limit(2 ГБ)` — потолок кэша (переопределяется `MLX_CACHE_GB`)
- `mx.clear_cache()` после каждой генерации — отдаёт буферы системе

С этим сервер держится ~2.9 ГБ в простое вместо 36 ГБ.

## Грабли

- **Русского нет в `LANGUAGE_TOKEN` порта.** В mlx-audio словарь
  `confucius4.LANGUAGE_TOKEN` — это подмножество (zh/en/vi/ja/ko/th), и
  `lang="ru"` молча падает на английский instruction-токен. `say.py` и
  `server.py` дошивают русский токен (`请用俄语朗读接下来的文字`) до генерации.
- Импортировать надо из submodule
  `mlx_audio.tts.models.confucius4.confucius4`, а не из пакета.
- `soundfile` не тянется как зависимость mlx-audio — ставим отдельно.
