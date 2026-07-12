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
GATE = float(os.environ.get("GATE", "1200"))   # пиковая амплитуда: тише — считаем тишиной
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
    print("=" * 60)
    print("  Детектор wake word «Кот, слушай»")
    print(f"  Микрофон: {sd.query_devices(kind='input')['name']} ({native_sr} Гц)")
    print(f"  Порог срабатывания: {TH}   |   Ctrl+C — выход")
    print("  Скажи «Кот, слушай» — увидишь ✅ РАСПОЗНАЛ")
    print("=" * 60)
    last_fire = 0.0
    count = 0
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

            peak = int(np.abs(buf).max())                 # громкость окна
            t0 = time.perf_counter()
            sc = score() if peak >= GATE else 0.0         # гейт: на тишине не считаем
            ms = (time.perf_counter() - t0) * 1000        # время обработки окна

            filled = int(sc * 30)
            bar = "█" * filled + "·" * (30 - filled)
            mark = "◄ РАСПОЗНАЮ" if sc >= TH else ""
            print(f"\rскор {sc:0.2f} [{bar}] {ms:4.0f}мс {mark:<12}", end="", flush=True)

            if sc >= TH and time.time() - last_fire > 1.5:
                last_fire = time.time()
                count += 1
                ts = time.strftime("%H:%M:%S")
                print(f"\n✅ [{ts}]  РАСПОЗНАЛ «Кот, слушай»   "
                      f"(уверенность {sc:.2f}, обработка {ms:.0f}мс, всего: {count})")
    except KeyboardInterrupt:
        print(f"\n\nвыход. Распознаваний за сессию: {count}")
        stream.stop(); stream.close()


if __name__ == "__main__":
    main()
