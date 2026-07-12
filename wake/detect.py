#!/usr/bin/env python3
"""
Живой детектор wake word «Кот, слушай»: микрофон -> AudioFeatures -> onnx -> порог.
Фичи считаются ровно как при обучении (скользящее 2-сек окно, embed_clips).

Запуск:   .venv/bin/python detect.py
Порог:    THRESHOLD=0.8 .venv/bin/python detect.py
"""
import os, sys, time
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from scipy.signal import resample_poly
import openwakeword.utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
TH = float(os.environ.get("THRESHOLD", "0.8"))
SR = 16000
WIN = 2 * SR          # окно 2 сек = (16,96)
HOP = 4000            # считаем скор каждые 0.25 сек

F = U.AudioFeatures(device="cpu")
sess = ort.InferenceSession(os.path.join(HERE, "kot_slushai.onnx"))
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name

buf = np.zeros(WIN, dtype=np.int16)


def score():
    f = F.embed_clips(buf[None, :], batch_size=1).astype(np.float32)
    return float(sess.run([oname], {iname: f})[0].flatten()[0])


def main():
    global buf
    # микрофон: пробуем 16кГц; если устройство не умеет — его частота + ресемплинг
    try:
        stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=HOP)
        stream.start()
        native_sr = SR
    except Exception:
        native_sr = int(sd.query_devices(kind="input")["default_samplerate"])
        blk = round(HOP * native_sr / SR)
        stream = sd.InputStream(samplerate=native_sr, channels=1, dtype="int16", blocksize=blk)
        stream.start()
    print(f"🎤 Слушаю ({native_sr} Гц). Скажи «Кот, слушай». Порог={TH}. Ctrl+C — выход.")
    last_fire = 0.0
    try:
        while True:
            data, _ = stream.read(stream.blocksize)
            chunk = data[:, 0].astype(np.float32)
            if native_sr != SR:                       # ресемплинг -> 16кГц
                chunk = resample_poly(chunk, SR, native_sr)
            chunk = chunk.astype(np.int16)
            n = len(chunk)
            buf = np.roll(buf, -n)
            buf[-n:] = chunk
            sc = score()
            bar = "█" * int(sc * 24)
            hit = "  🐱 КОТ УСЛЫШАЛ!" if sc >= TH else ""
            print(f"\r{sc:0.2f} |{bar:<24}|{hit}   ", end="", flush=True)
            if sc >= TH and time.time() - last_fire > 1.5:
                last_fire = time.time()
                print(f"\n>>> СРАБОТАЛО ({sc:.2f}) <<<")
    except KeyboardInterrupt:
        print("\nвыход")
        stream.stop(); stream.close()


if __name__ == "__main__":
    main()
