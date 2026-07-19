#!/usr/bin/env python3
"""Тесты модели с нуля на тех же наборах, что и ruBERT.

Оба набора held-out для обеих моделей (train_tiny и train_joint вычитают одно
и то же), поэтому сравнение честное: разница в цифрах — это разница моделей,
а не данных. Контракт у обеих одинаковый: список команд.

    cd mlx-chat && .venv/bin/python ft/test_tiny.py
"""
import os
import time

import test_joint
import test_whisper

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    return None if s is None else s.lower().replace("-", " ").replace(".", "").strip()


def run(predict, label):
    predict("прогрев")

    # одиночные команды
    t0 = time.time()
    single = [predict(t) for t, _, _ in test_joint.CASES]
    dt = (time.time() - t0) / len(test_joint.CASES)
    ok_s, fails_s = 0, []
    for (text, ei, es), got in zip(test_joint.CASES, single):
        expected = test_joint.as_expected(ei, es)
        if test_joint.cmp_commands(got, expected):
            ok_s += 1
        else:
            fails_s.append((text, expected, got))

    # составные
    ok_m, fails_m = 0, []
    for text, expected in test_joint.MULTI_CASES:
        got = predict(text)
        if test_joint.cmp_commands(got, expected):
            ok_m += 1
        else:
            fails_m.append((text, expected, got))

    # живые расшифровки Whisper: проверяем интент и пару (ticket, num)
    ok_w, fails_w = 0, []
    for text, ei, et, en in test_whisper.CASES:
        cmds = predict(text)
        if ei == "none":
            good = len(cmds) == 0
            gi, gt, gn = ("none" if not cmds else f"{len(cmds)} команд"), None, None
        elif len(cmds) == 1:
            gi = cmds[0]["intent"]
            gt, gn = cmds[0]["slots"].get("ticket"), cmds[0]["slots"].get("num")
            good = gi == ei and norm(gt) == norm(et) and norm(gn) == norm(en)
        else:
            gi, gt, gn = f"{len(cmds)} команд", None, None
            good = False
        ok_w += good
        if not good:
            fails_w.append((text, f"{ei} {et}/{en}", f"{gi} {gt}/{gn}"))

    n_s, n_m, n_w = len(test_joint.CASES), len(test_joint.MULTI_CASES), len(test_whisper.CASES)
    print(f"=== {label} ===")
    print(f"  одиночные:     {ok_s}/{n_s}")
    print(f"  составные:     {ok_m}/{n_m}")
    print(f"  живые Whisper: {ok_w}/{n_w}")
    print(f"  скорость:      {dt*1000:.2f} мс/запрос (CPU)")
    for text, expected, got in fails_s + fails_m:
        print(f"    «{text}»")
        print(f"        ждали {expected}")
        print(f"        вышло {[(g['intent'], g['slots']) for g in got]}")
    for text, exp, got in fails_w:
        print(f"    [whisper] «{text}»  ждали {exp}  вышло {got}")


def main():
    import tiny_model_def as tm
    model, cfg = tm.load(os.path.join(HERE, "tiny_model"))
    n = sum(p.numel() for p in model.parameters())
    size = sum(os.path.getsize(os.path.join(HERE, "tiny_model", f))
               for f in os.listdir(os.path.join(HERE, "tiny_model")))
    run(lambda t: tm.predict(model, cfg, t),
        f"С нуля — {n/1000:.0f}K параметров, {size/1024/1024:.1f} МБ на диске")


if __name__ == "__main__":
    main()
