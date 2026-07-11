#!/usr/bin/env python3
"""Клиент к F5-TTS server.py: текст -> .wav без перезагрузки модели.

    python say.py "Текст голосом Lily"
    python say.py "Текст" --out out/my.wav --nfe 16 --speed 1.0
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

URL = "http://127.0.0.1:8124/gen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    data = {"text": a.text, "nfe": str(a.nfe), "speed": str(a.speed)}
    if a.out:
        data["out"] = a.out
    body = urllib.parse.urlencode(data).encode()
    try:
        with urllib.request.urlopen(URL, data=body, timeout=600) as r:
            res = json.load(r)
    except urllib.error.URLError as e:
        sys.exit(f"Сервер не отвечает ({e}). Запусти: python server.py")

    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not a.no_open and res.get("out"):
        # afplay — системный плеер, без диалога Gatekeeper (в отличие от `open`)
        subprocess.run(["afplay", res["out"]], check=False)


if __name__ == "__main__":
    main()
