#!/usr/bin/env python3
"""
Генерация позитивного датасета для wake word «Кот, слушай» через ElevenLabs.

Берём родные русские голоса из shared-библиотеки ElevenLabs (много разных
дикторов, муж/жен) + русские голоса аккаунта, каждым озвучиваем фразу в
нескольких вариантах просодии. Сохраняем сразу в 16 кГц моно 16-бит WAV —
ровно тот формат, который ждёт обучалка openWakeWord.

Негативы генерить не нужно: в Colab openWakeWord подмешивает большой корпус
негативов (ACAV100M) и сам делает аугментацию (шум/реверберация). Наша задача —
дать разнообразные ЧИСТЫЕ позитивы.

Зависимостей нет (только стандартная библиотека). Ключ читается из wake/.env.

Использование:
  python3 gen_dataset.py                # дефолт: ~40 голосов, все фразы
  VOICES=60 python3 gen_dataset.py      # больше голосов
  python3 gen_dataset.py --dry-run      # только показать список голосов
"""
import json
import os
import sys
import time
import wave
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "positives")
MODEL_ID = "eleven_multilingual_v2"          # вторая модель, как просил пользователь
API = "https://api.elevenlabs.io"

# Варианты произношения фразы — пунктуация меняет интонацию/паузу.
PHRASES = [
    "Кот, слушай",
    "Кот, слушай!",
    "Кот, слушай.",
    "Кот... слушай",
    "Кот, слушай?",
]

# Разные настройки голоса → разброс по выразительности/темпу.
SETTINGS = [
    {"stability": 0.35, "similarity_boost": 0.75, "style": 0.0},
    {"stability": 0.55, "similarity_boost": 0.80, "style": 0.30},
    {"stability": 0.75, "similarity_boost": 0.75, "style": 0.0},
]

N_VOICES = int(os.environ.get("VOICES", "40"))
WORKERS = int(os.environ.get("WORKERS", "5"))   # параллельные запросы к ElevenLabs
# сколько вариантов фраз/настроек использовать (для маленьких пробных прогонов)
N_PHRASES = int(os.environ.get("PHRASES_N", str(len(PHRASES))))
N_SETTINGS = int(os.environ.get("SETTINGS_N", str(len(SETTINGS))))
PHRASES = PHRASES[:N_PHRASES]
SETTINGS = SETTINGS[:N_SETTINGS]


def load_key():
    env = os.path.join(HERE, ".env")
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line.startswith("ELEVENLABS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("ELEVENLABS_API_KEY", "")


KEY = load_key()
if not KEY:
    sys.exit("Нет ELEVENLABS_API_KEY (положи в wake/.env)")


def req(url, data=None, headers=None, method=None):
    h = {"xi-api-key": KEY}
    if headers:
        h.update(headers)
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=h, method=method)
    return urllib.request.urlopen(r, timeout=60)


def collect_voices():
    """Родные русские голоса: shared-библиотека + голоса аккаунта."""
    voices = []
    seen = set()
    # 1) shared-библиотека, язык ru
    try:
        with req(f"{API}/v1/shared-voices?page_size={max(N_VOICES,40)}&language=ru") as r:
            for v in json.load(r).get("voices", []):
                vid = v.get("voice_id")
                if vid and vid not in seen:
                    seen.add(vid)
                    voices.append((vid, v.get("name", "?"), v.get("gender", "?")))
    except urllib.error.HTTPError as e:
        print("shared-voices error:", e.read().decode()[:200])
    # 2) русские голоса из аккаунта (label language == ru)
    try:
        with req(f"{API}/v2/voices?page_size=100") as r:
            for v in json.load(r).get("voices", []):
                if v.get("labels", {}).get("language") == "ru":
                    vid = v["voice_id"]
                    if vid not in seen:
                        seen.add(vid)
                        voices.append((vid, v.get("name", "?"), v.get("labels", {}).get("gender", "?")))
    except urllib.error.HTTPError as e:
        print("account voices error:", e.read().decode()[:200])
    return voices[:N_VOICES]


def pcm16k_to_wav(pcm_bytes, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)        # 16-bit
        w.setframerate(16000)
        w.writeframes(pcm_bytes)


def synth(voice_id, text, settings):
    url = f"{API}/v1/text-to-speech/{voice_id}?output_format=pcm_16000"
    payload = {"text": text, "model_id": MODEL_ID, "voice_settings": settings}
    with req(url, data=payload) as r:
        return r.read()


def main():
    dry = "--dry-run" in sys.argv
    voices = collect_voices()
    print(f"Голосов собрано: {len(voices)}")
    for vid, name, g in voices:
        print(f"  {g:6} | {name[:40]:40} | {vid}")
    if dry:
        return

    os.makedirs(OUT, exist_ok=True)

    # Собираем список заданий (пропускаем уже готовые файлы).
    jobs = []
    for vi, (vid, name, g) in enumerate(voices):
        for pi, phrase in enumerate(PHRASES):
            for si, st in enumerate(SETTINGS):
                fname = f"kot_{vi:02d}_{pi}_{si}.wav"
                fpath = os.path.join(OUT, fname)
                if os.path.exists(fpath):
                    continue
                jobs.append((fname, fpath, vid, name, g, phrase, st))

    total = len(jobs)
    print(f"К генерации: {total} клипов, потоков: {WORKERS}")
    manifest = []
    lock = threading.Lock()
    done = [0]
    t0 = time.monotonic()

    def work(job):
        fname, fpath, vid, name, g, phrase, st = job
        for attempt in range(5):
            try:
                pcm = synth(vid, phrase, st)
                pcm16k_to_wav(pcm, fpath)
                with lock:
                    manifest.append({"file": fname, "voice": vid, "name": name,
                                     "gender": g, "phrase": phrase, "settings": st})
                    done[0] += 1
                    if done[0] % 20 == 0:
                        print(f"  {done[0]}/{total}  ({time.monotonic()-t0:.0f}s)")
                return
            except urllib.error.HTTPError as e:
                if e.code == 429:                      # rate limit → отступаем
                    time.sleep(1 + attempt * 2)
                else:
                    print(f"  ! {fname}: HTTP {e.code} {e.read().decode()[:100]}")
                    time.sleep(0.5)
            except Exception as e:
                print(f"  ! {fname}: {e}")
                time.sleep(0.5)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, jobs))

    # дописываем к существующему манифесту, если он есть
    mpath = os.path.join(HERE, "out", "manifest.json")
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath))
            have = {m["file"] for m in manifest}
            manifest = [m for m in old if m["file"] not in have] + manifest
        except Exception:
            pass
    with open(mpath, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\nГотово: {len(manifest)} WAV в {OUT}")
    print(f"Манифест: {os.path.join(HERE, 'out', 'manifest.json')}")


if __name__ == "__main__":
    main()
