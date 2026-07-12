#!/usr/bin/env python3
"""
Живой детектор wake word «Эй, кот»: микрофон -> AudioFeatures -> onnx -> порог.
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
TH = float(os.environ.get("THRESHOLD", "0.85"))
GATE = float(os.environ.get("GATE", "1200"))   # пиковая амплитуда: тише — считаем тишиной
CONSEC = int(os.environ.get("CONSEC", "2"))    # сколько окон подряд выше порога для срабатывания
SR = 16000
WIN = 2 * SR          # окно 2 сек = (16,96)
HOP = 4000            # считаем скор каждые 0.25 сек

F = U.AudioFeatures(device="cpu")
sess = ort.InferenceSession(os.path.join(HERE, "kot_slushai.onnx"))
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name

buf = np.zeros(WIN, dtype=np.int16)


def score():
    # ВАЖНО: _get_embeddings детерминирован; F.embed_clips (батч) — НЕТ (np.empty-баг)
    e = F._get_embeddings(buf.astype(np.int16))[:16]        # (16,96)
    f = e[None, :].astype(np.float32)
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
    print("  Детектор wake word «Эй, кот»")
    print(f"  Микрофон: {sd.query_devices(kind='input')['name']} ({native_sr} Гц)")
    print(f"  Порог срабатывания: {TH}   |   Ctrl+C — выход")
    print("  Скажи «Эй, кот» — увидишь ✅ РАСПОЗНАЛ")
    print("=" * 60)
    last_fire = 0.0
    count = 0
    consec = 0                                            # подряд окон выше порога
    frame = 0
    IS_TTY = sys.stdout.isatty()
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

            consec = consec + 1 if sc >= TH else 0        # сглаживание: считаем подряд
            frame += 1
            if IS_TTY:                                    # живой индикатор (настоящий терминал)
                filled = int(sc * 30)
                bar = "█" * filled + "·" * (30 - filled)
                vol = "🔊" if peak >= GATE else "· "       # микрофон принимает сигнал?
                mark = f"◄ {consec}/{CONSEC}" if sc >= TH else ""
                print(f"\rскор {sc:0.2f} [{bar}] {vol}вход {peak:5d} {ms:4.0f}мс {mark:<10}",
                      end="", flush=True)
            elif frame % 4 == 0 or sc >= TH:              # без TTY — обычными строками
                print(f"скор {sc:0.2f}  peak={peak:5d}  {ms:3.0f}мс"
                      + (f"  ◄ {consec}/{CONSEC}" if sc >= TH else ""), flush=True)

            # срабатываем только если порог держится CONSEC окон подряд (режет случайные всплески)
            if consec >= CONSEC and time.time() - last_fire > 1.5:
                last_fire = time.time()
                count += 1
                ts = time.strftime("%H:%M:%S")
                print(f"\n✅ [{ts}]  РАСПОЗНАЛ «Эй, кот»   "
                      f"(уверенность {sc:.2f}, обработка {ms:.0f}мс, всего: {count})")
    except KeyboardInterrupt:
        print(f"\n\nвыход. Распознаваний за сессию: {count}")
        stream.stop(); stream.close()


if __name__ == "__main__":
    main()
