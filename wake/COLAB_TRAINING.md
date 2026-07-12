# Обучение wake word «Кот, слушай» в Colab (openWakeWord)

Идея: используем **официальный** ноутбук openWakeWord как основу, но подменяем
его английские piper-позитивы на **наши 80 русских клипов** из ElevenLabs
(`wake/out/kot_positives.zip`). Всё остальное (негативы ACAV100M, шумы, RIR,
аугментация, обучение головы, экспорт `.onnx`) делает ноутбук.

Негативы руками НЕ готовим — их приносит сам ноутбук (большой корпус + шумы).

---

## 0. Открыть ноутбук

Официальный «automatic training»:
https://github.com/dscripka/openWakeWord → `notebooks/automatic_model_training.ipynb`
→ кнопка **Open in Colab**. Runtime → Change runtime type → **GPU** (T4 хватает).

## 1. Прогнать как есть до генерации клипов

Выполни ячейки по порядку:
- **Setup** (clone repos, `pip install ...`) — как в ноутбуке.
- **Download data** — качает RIR (MIT), шумы (AudioSet/FMA), фичи негативов
  (`davidscripka/openwakeword_features`, ~2000 ч ACAV100M) и валидацию.

## 2. Задать нашу фразу в конфиге

В ячейке, где правится конфиг (`config[...] = ...`), выставь:

```python
config["target_phrase"]  = ["кот слушай"]      # список
config["model_name"]     = "kot_slushai"
config["n_samples"]      = 2000                 # ноутбук сгенерит СВОИ (англ.) — мы их перезапишем
config["n_samples_val"]  = 200
config["steps"]          = 10000                # для первой пробы норм
config["target_accuracy"]= 0.7
config["target_recall"]  = 0.5
```

`output_dir` оставь дефолтным (обычно `./my_custom_model` — запомни значение,
оно понадобится в шаге 4).

## 3. Запустить генерацию клипов (ради структуры и НЕГАТИВОВ)

```bash
!python openwakeword/train.py --training_config custom_model.yml --generate_clips
```

Это создаст папки:
```
{output_dir}/kot_slushai/positive_train/   ← сюда лягут англ. позитивы (выкинем)
{output_dir}/kot_slushai/positive_test/    ← и сюда (выкинем)
{output_dir}/kot_slushai/negative_train/   ← негативы (ОСТАВЛЯЕМ)
{output_dir}/kot_slushai/negative_test/    ← (ОСТАВЛЯЕМ)
```

## 4. Подменить позитивы на НАШИ русские клипы

Загрузи `kot_positives.zip` (слева «Files» → Upload) и выполни ячейку
(поправь `OUT` под свой `output_dir`, если он не `my_custom_model`):

```python
import os, glob, zipfile, random, shutil

OUT = "my_custom_model/kot_slushai"          # = {output_dir}/{model_name}
ptrain = os.path.join(OUT, "positive_train")
ptest  = os.path.join(OUT, "positive_test")

# распаковать наш архив
with zipfile.ZipFile("kot_positives.zip") as z:
    z.extractall("kot_src")
wavs = sorted(glob.glob("kot_src/positives/*.wav"))
print("наших клипов:", len(wavs))

# очистить сгенерённые англ. позитивы
for d in (ptrain, ptest):
    for f in glob.glob(os.path.join(d, "*.wav")):
        os.remove(f)

# сплит 90/10 в train/test
random.seed(0); random.shuffle(wavs)
k = max(1, int(len(wavs) * 0.1))
test, train = wavs[:k], wavs[k:]
for f in train: shutil.copy(f, ptrain)
for f in test:  shutil.copy(f, ptest)
print("positive_train:", len(os.listdir(ptrain)), "| positive_test:", len(os.listdir(ptest)))
```

> ⚠️ 80 клипов — маловато (openWakeWord любит тысячи). Для ПЕРВОЙ пробы норм:
> аугментация (шаг 5) размножит их шумом/реверберацией. Если детектор выйдет
> слабым — вернёмся и нагенерим в ElevenLabs 300–500 клипов (это быстро).

## 5. Аугментация (уже с нашими позитивами)

```bash
!python openwakeword/train.py --training_config custom_model.yml --augment_clips
```

## 6. Обучение

```bash
!python openwakeword/train.py --training_config custom_model.yml --train_model
```

## 7. Скачать модель

Результат: `{output_dir}/kot_slushai.onnx` (и `.tflite`, он нам не нужен).
Скачай `kot_slushai.onnx` к себе → положим в `wake/` и на нём построим локальный
детектор (микрофон → `onnxruntime` → пикнул при «Кот, слушай»).

---

### Если что-то отвалится
- Названия ячеек/конфига в ноутбуке могут чуть отличаться — ориентируйся на
  ключи `target_phrase`, `model_name`, `output_dir`, `--generate_clips/--augment_clips/--train_model`.
- Проверь фактический `output_dir` в конфиге и подставь его в шаг 4.
