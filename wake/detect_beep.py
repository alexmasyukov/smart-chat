#!/usr/bin/env python3
"""
Живой детектор «Эй, кот» с ЗВУКОВЫМ сигналом при срабатывании.
Висит, слушает микрофон. Как распознал — играет бип (afplay, macOS) и печатает,
за сколько сработал (латентность от начала фразы + время обработки окна).

Запуск:   .venv/bin/python detect_beep.py
Крутилки: THRESHOLD=0.9 GATE=1500 CONSEC=2 SOUND=Glass .venv/bin/python detect_beep.py
          SOUND=- .venv/bin/python detect_beep.py   # выключить звук
"""
import os, sys, time, subprocess
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from scipy.signal import resample_poly
import openwakeword.utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
TH = float(os.environ.get("THRESHOLD", "0.85"))
GATE = float(os.environ.get("GATE", "1200"))    # пиковая амплитуда: тише — тишина
CONSEC = int(os.environ.get("CONSEC", "2"))      # окон подряд выше порога для срабатывания
SOUND = os.environ.get("SOUND", "Glass")         # имя системного звука macOS ('-' = без звука)
SR = 16000
WIN = 2 * SR          # окно 2 сек = (16,96)
HOP = int(os.environ.get("HOP", "4000"))   # шаг проверки в сэмплах (4000=0.25с). Меньше = быстрее реакция, больше CPU

F = U.AudioFeatures(device="cpu")
sess = ort.InferenceSession(os.path.join(HERE, "kot_slushai.onnx"))
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name

buf = np.zeros(WIN, dtype=np.int16)


def beep():
    """Неблокирующий звуковой сигнал (macOS afplay системного звука)."""
    if SOUND == "-":
        print("\a", end="", flush=True)            # терминальный bell как запасной
        return SOUND
    path = f"/System/Library/Sounds/{SOUND}.aiff"
    try:
        subprocess.Popen(["afplay", path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"{SOUND}.aiff"
    except Exception:
        print("\a", end="", flush=True)
        return "bell"


def score():
    # _get_embeddings детерминирован; F.embed_clips (батч) — НЕТ (np.empty-баг)
    e = F._get_embeddings(buf.astype(np.int16))[:16]        # (16,96)
    f = e[None, :].astype(np.float32)
    return float(sess.run([oname], {iname: f})[0].flatten()[0])


def main():
    global buf
    try:
        stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=HOP)
        stream.start()
        native_sr = SR
    except Exception:
        native_sr = int(sd.query_devices(kind="input")["default_samplerate"])
        blk = round(HOP * native_sr / SR)
        stream = sd.InputStream(samplerate=native_sr, channels=1, dtype="int16", blocksize=blk)
        stream.start()
    print("=" * 62)
    print("  Детектор «Эй, кот» — только бипы (без индикатора)")
    print(f"  Микрофон: {sd.query_devices(kind='input')['name']} ({native_sr} Гц)")
    print(f"  Порог {TH} | гейт {GATE} | consec {CONSEC} | сигнал: {SOUND}")
    print("  Скажи «Эй, кот» — услышишь бип. Ctrl+C — выход")
    print("=" * 62, flush=True)
    last_fire = 0.0
    count = 0
    consec = 0
    speech_start = None       # когда громкость впервые поднялась выше гейта (начало фразы)
    try:
        while True:
            data, _ = stream.read(stream.blocksize)
            chunk = data[:, 0].astype(np.float32)
            if native_sr != SR:
                chunk = resample_poly(chunk, SR, native_sr)
            chunk = chunk.astype(np.int16)
            n = len(chunk)
            buf = np.roll(buf, -n)
            buf[-n:] = chunk

            peak = int(np.abs(buf).max())
            loud = peak >= GATE
            now = time.time()
            if loud and speech_start is None:
                speech_start = now                 # засекли начало звука
            elif not loud:
                speech_start = None                # тишина — сброс

            t0 = time.perf_counter()
            sc = score() if loud else 0.0
            ms = (time.perf_counter() - t0) * 1000

            consec = consec + 1 if sc >= TH else 0

            if consec >= CONSEC and now - last_fire > 1.5:
                last_fire = now
                count += 1
                snd = beep()                        # ← ЗВУКОВОЙ СИГНАЛ
                # латентность от начала фразы до срабатывания
                lat = (now - speech_start) * 1000 if speech_start else float("nan")
                ts = time.strftime("%H:%M:%S")
                print(f"\n✅ [{ts}]  РАСПОЗНАЛ «Эй, кот»  🔔 сигнал: {snd}")
                print(f"   уверенность {sc:.2f} | реакция от начала фразы "
                      f"{lat:.0f}мс | обработка окна {ms:.0f}мс | всего: {count}\n")
                consec = 0
                speech_start = None
    except KeyboardInterrupt:
        print(f"\n\nвыход. Распознаваний за сессию: {count}")
        stream.stop(); stream.close()


if __name__ == "__main__":
    main()
