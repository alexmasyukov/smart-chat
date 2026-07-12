#!/usr/bin/env python3
"""
Генерация НЕГАТИВОВ для wake word «Кот, слушай» через ElevenLabs.

Нужна разнообразная русская речь, которая НЕ содержит «кот, слушай», плюс
фонетически близкие хард-негативы (кот, кто, слушай, компот…). Фразы длинные —
при обучении нарежем их скользящим окном 2 сек → много негативных окон.

Пишет 16 кГц моно WAV в out/negatives/. Ключ из wake/.env.
"""
import json, os, sys, time, wave, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "negatives")
API = "https://api.elevenlabs.io"
MODEL_ID = "eleven_multilingual_v2"
N_VOICES = int(os.environ.get("VOICES", "10"))
WORKERS = int(os.environ.get("WORKERS", "3"))

# Длинные нейтральные фразы (речь-негатив).
SENTENCES = [
    "Сегодня прекрасная погода, и я решил прогуляться по парку недалеко от дома.",
    "Мне нужно купить хлеб, молоко и немного овощей на ужин для всей семьи.",
    "Вчера мы смотрели интересный фильм, а потом долго обсуждали его за чаем.",
    "Пожалуйста, открой окно, в комнате стало очень душно и жарко.",
    "Завтра утром у меня важная встреча, поэтому лягу спать пораньше.",
    "Он поставил чайник, достал чашки и начал готовить завтрак для гостей.",
    "Эта книга оказалась намного увлекательнее, чем я ожидал вначале.",
    "Дети играли во дворе, смеялись и бегали до самого вечера.",
    "Включи, пожалуйста, музыку погромче, мне нравится эта старая песня.",
    "Мы поехали за город, чтобы отдохнуть у реки и пожарить шашлык.",
    "Никак не могу найти свои ключи, кажется, я оставил их на работе.",
    "Расскажи мне, как прошёл твой день, что интересного случилось.",
    "На завтрак я обычно ем овсяную кашу с фруктами и пью крепкий кофе.",
    "Собака радостно виляла хвостом, встречая хозяина у порога.",
    "Через час начнётся дождь, не забудь взять с собой зонт.",
]
SENTENCES += [
    "Интересно, что будет показывать телевизор сегодня поздно вечером.",
    "Я думаю, нам стоит перекрасить стены в спальне в светлый оттенок.",
    "Поезд прибывает ровно в полдень, поэтому нужно выйти заранее.",
    "Кажется, где-то в квартире протекает кран, слышно постоянное капанье.",
    "Давай закажем пиццу и посмотрим какой-нибудь новый сериал вместе.",
    "Утром я пробежал пять километров и чувствую себя прекрасно.",
    "На столе лежат documents, ручка и чашка недопитого кофе.",
    "Соседи снова затеяли ремонт, весь день слышно дрель и молоток.",
    "Мне кажется, эта картина отлично впишется в интерьер гостиной.",
    "Пора собирать чемодан, самолёт вылетает завтра рано утром.",
    "Он долго объяснял, как правильно настроить новый роутер дома.",
    "Осенью в лесу очень красиво, повсюду жёлтые и красные листья.",
    "Проверь, пожалуйста, выключил ли ты плиту перед уходом из дома.",
    "Мой любимый напиток — зелёный чай с мятой и небольшим количеством меда.",
    "Вечером обещали сильный ветер, лучше закрыть все окна поплотнее.",
]
# Короткие фонетически близкие хард-негативы.
HARD = ["кот", "кто", "код", "компот", "котик", "слушай", "слушать", "послушай",
        "слушай меня внимательно", "кот спит на диване", "кот, привет", "который час",
        "скотч", "котлета", "кода нет", "кит", "рот", "вот", "крот", "поток",
        "слушатель", "слушаю тебя", "хорошо слушай сюда", "кот и собака", "мокрый кот",
        "нет", "да", "привет", "спасибо", "погоди", "стоп", "окей", "дальше", "поехали"]

PHRASES = SENTENCES + HARD


def load_key():
    for line in open(os.path.join(HERE, ".env")):
        if line.startswith("ELEVENLABS_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


KEY = load_key() or sys.exit("нет ключа")


def req(url, data=None):
    h = {"xi-api-key": KEY}
    body = None
    if data is not None:
        body = json.dumps(data).encode(); h["Content-Type"] = "application/json"
    return urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h), timeout=90)


def voices():
    out = []
    with req(f"{API}/v1/shared-voices?page_size={N_VOICES}&language=ru") as r:
        for v in json.load(r).get("voices", [])[:N_VOICES]:
            out.append(v["voice_id"])
    return out


def to_wav(pcm, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)


def synth(vid, text):
    url = f"{API}/v1/text-to-speech/{vid}?output_format=pcm_16000"
    with req(url, {"text": text, "model_id": MODEL_ID}) as r:
        return r.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    vs = voices()
    print(f"голосов: {len(vs)}, фраз: {len(PHRASES)}")
    jobs = [(pi, phrase, vi, vid) for pi, phrase in enumerate(PHRASES) for vi, vid in enumerate(vs)]
    done = [0]; lock = threading.Lock()

    def work(job):
        pi, phrase, vi, vid = job
        fp = os.path.join(OUT, f"neg_{pi:02d}_{vi:02d}.wav")
        if os.path.exists(fp):
            return
        for attempt in range(5):
            try:
                to_wav(synth(vid, phrase), fp)
                with lock:
                    done[0] += 1
                    if done[0] % 20 == 0:
                        print(f"  {done[0]}/{len(jobs)}")
                return
            except urllib.error.HTTPError as e:
                time.sleep(1 + attempt * 2 if e.code == 429 else 0.5)
            except Exception:
                time.sleep(0.5)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, jobs))
    n = len([f for f in os.listdir(OUT) if f.endswith(".wav")])
    print(f"Готово: {n} негативных WAV в {OUT}")


if __name__ == "__main__":
    main()
