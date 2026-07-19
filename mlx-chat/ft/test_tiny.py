#!/usr/bin/env python3
"""Сравнение модели с нуля и ruBERT на ОДНИХ И ТЕХ ЖЕ наборах.

Оба набора held-out для обеих моделей (train_tiny и train_joint вычитают одно
и то же), поэтому сравнение честное: разница в цифрах — это разница моделей,
а не данных.

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
    """predict: (text) → (intent, slots, score). Возвращает строку отчёта."""
    predict("прогрев")

    # held-out (test_joint)
    t0 = time.time()
    hj = [predict(t)[:2] for t, _, _ in test_joint.CASES]
    dt_j = (time.time() - t0) / len(test_joint.CASES)
    ok_i = ok_s = 0
    fails_j = []
    for (text, ei, es), (gi, gs_) in zip(test_joint.CASES, hj):
        i_ok = gi == ei
        s_ok = es is None or ({k: norm(v) for k, v in gs_.items()} ==
                              {k: norm(v) for k, v in es.items()})
        ok_i += i_ok
        ok_s += s_ok
        if not (i_ok and s_ok):
            fails_j.append((text, ei, es, gi, gs_))

    # живые расшифровки Whisper
    t0 = time.time()
    hw = [predict(t) for t, _, _, _ in test_whisper.CASES]
    dt_w = (time.time() - t0) / len(test_whisper.CASES)
    ok_w = 0
    fails_w = []
    for (text, ei, et, en), (gi, gs_, _) in zip(test_whisper.CASES, hw):
        gt, gn = gs_.get("ticket"), gs_.get("num")
        good = gi == ei and norm(gt) == norm(et) and norm(gn) == norm(en)
        ok_w += good
        if not good:
            fails_w.append((text, ei, et, en, gi, gt, gn))

    n_j, n_w = len(test_joint.CASES), len(test_whisper.CASES)
    print(f"=== {label} ===")
    print(f"  held-out интент: {ok_i}/{n_j}   слоты: {ok_s}/{n_j}")
    print(f"  живые Whisper:   {ok_w}/{n_w}")
    print(f"  скорость:        {(dt_j+dt_w)/2*1000:.2f} мс/запрос (CPU)")
    for text, ei, es, gi, gs_ in fails_j:
        print(f"    [held-out] «{text}»  ждали {ei} {es}  вышло {gi} {gs_}")
    for text, ei, et, en, gi, gt, gn in fails_w:
        print(f"    [whisper] «{text}»  ждали {ei} {et}/{en}  вышло {gi} {gt}/{gn}")
    return ok_i + ok_s + ok_w


def main():
    import tiny_model_def as tm
    model, cfg = tm.load(os.path.join(HERE, "tiny_model"))
    n = sum(p.numel() for p in model.parameters())
    run(lambda t: tm.predict(model, cfg, t), f"С нуля ({n/1000:.0f}K параметров)")

    print()
    joint_dir = os.path.join(HERE, "joint_model")
    if os.path.isdir(joint_dir):
        import joint_model_def as jm
        jmodel, tok, jcfg = jm.load(joint_dir)
        jn = sum(p.numel() for p in jmodel.parameters())
        run(lambda t: jm.predict(jmodel, tok, jcfg, t), f"ruBERT ({jn/1e6:.0f}M параметров)")
    else:
        print("(joint_model нет — сравнить не с чем)")


if __name__ == "__main__":
    main()
