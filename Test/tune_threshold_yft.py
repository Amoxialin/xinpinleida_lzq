#!/usr/bin/env python
# coding: utf-8
"""Sweep the multi-label decision threshold on the dev split and report the
test performance at the threshold chosen on dev.

Usage:
    python3 tune_threshold_yft.py conf/train.yft.json
"""

import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
from model.classification.textcnn import TextCNN


def build_loader(conf, json_files):
    dataset = ClassificationDataset(conf, json_files)
    collate_fn = ClassificationCollator(conf, len(dataset.label_map))
    loader = DataLoader(dataset, batch_size=conf.eval.batch_size, shuffle=False,
                        num_workers=conf.data.num_worker,
                        collate_fn=collate_fn, pin_memory=True)
    return dataset, loader


def predict_probs(conf, model, loader):
    probs, golds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            golds.extend(batch[ClassificationDataset.DOC_LABEL_LIST])
    return np.asarray(probs, dtype=np.float32), golds


def metrics(probs, golds, threshold, top_k, num_labels):
    """Return micro/macro P,R,F1, exact match ratio and per-bucket F1."""
    order = np.argsort(-probs, axis=1)
    right = np.zeros(num_labels)
    predict = np.zeros(num_labels)
    standard = np.zeros(num_labels)
    exact = 0
    for i, prob in enumerate(probs):
        idx = order[i][:top_k]
        idx = idx[prob[idx] > threshold]
        pred = set(int(x) for x in idx)
        gold = set(int(x) for x in golds[i])
        for p in pred:
            predict[p] += 1
        for g in gold:
            standard[g] += 1
        for p in pred & gold:
            right[p] += 1
        if pred == gold:
            exact += 1

    def prf(r, p, s):
        prec = r / p if p > 0 else 0.0
        rec = r / s if s > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        return prec, rec, f

    micro = prf(right.sum(), predict.sum(), standard.sum())
    supported = standard > 0
    per_label_f = np.zeros(num_labels)
    for j in range(num_labels):
        per_label_f[j] = prf(right[j], predict[j], standard[j])[2]
    macro = per_label_f[supported].mean() if supported.any() else 0.0
    return {
        "micro_p": micro[0], "micro_r": micro[1], "micro_f1": micro[2],
        "macro_f1": float(macro),
        "exact_match": exact / len(probs),
        "per_label_f1": per_label_f, "support": standard,
        "num_predicted": int(predict.sum()), "num_gold": int(standard.sum()),
    }


def bucket_report(res, freq_train, num_labels):
    buckets = [("head(>=100)", 100, 10 ** 9),
               ("torso(10-99)", 10, 100),
               ("tail(<10)", 0, 10)]
    out = []
    supported = res["support"] > 0
    for name, low, high in buckets:
        mask = supported & (freq_train >= low) & (freq_train < high)
        if mask.sum() == 0:
            out.append((name, 0, 0.0))
            continue
        out.append((name, int(mask.sum()),
                    float(res["per_label_f1"][mask].mean())))
    return out


def main():
    conf = Config(config_file=sys.argv[1])
    num_labels = None

    dev_dataset, dev_loader = build_loader(conf, conf.data.validate_json_files)
    test_dataset, test_loader = build_loader(conf, conf.data.test_json_files)
    num_labels = len(dev_dataset.label_map)
    id_to_label = {v: k for k, v in dev_dataset.label_map.items()}

    empty_dataset = ClassificationDataset(conf, [])
    model = TextCNN(empty_dataset, conf)
    if conf.device.startswith("cuda"):
        model = model.cuda(conf.device)
    ckpt = torch.load(conf.eval.model_dir, map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("loaded checkpoint epoch=%s best_performance=%.6f"
          % (ckpt.get("epoch"), ckpt.get("best_performance", float("nan"))))

    dev_probs, dev_golds = predict_probs(conf, model, dev_loader)
    test_probs, test_golds = predict_probs(conf, model, test_loader)

    # train label frequency, used only for bucketed reporting
    freq_train = np.zeros(num_labels)
    with open(conf.data.train_json_files[0], encoding="utf-8") as fin:
        for line in fin:
            for label in json.loads(line)["doc_label"]:
                if label in dev_dataset.label_map:
                    freq_train[dev_dataset.label_map[label]] += 1

    print("num_labels=%d dev=%d test=%d" % (num_labels, len(dev_probs),
                                            len(test_probs)))
    results = []
    for top_k in (1, 2, 3, 5, 8):
        for thr in np.arange(0.05, 0.91, 0.05):
            d = metrics(dev_probs, dev_golds, thr, top_k, num_labels)
            t = metrics(test_probs, test_golds, thr, top_k, num_labels)
            results.append((float(thr), top_k, d, t))

    best_micro = max(results, key=lambda r: r[2]["micro_f1"])
    best_macro = max(results, key=lambda r: r[2]["macro_f1"])
    print("best by DEV micro-F1: thr=%.2f top_k=%d (devF1=%.4f)"
          % (best_micro[0], best_micro[1], best_micro[2]["micro_f1"]))
    print("best by DEV macro-F1: thr=%.2f top_k=%d (devMacro=%.4f)"
          % (best_macro[0], best_macro[1], best_macro[2]["macro_f1"]))

    print("\nsweep at top_k=%d:" % best_micro[1])
    print("%-5s | %-31s | %-31s" % ("thr", "DEV  microP microR microF1 macroF1",
                                    "TEST microP microR microF1 macroF1"))
    for thr, top_k, d, t in results:
        if top_k != best_micro[1]:
            continue
        print("%-5.2f | %.4f  %.4f  %.4f  %.4f | %.4f  %.4f  %.4f  %.4f"
              % (thr, d["micro_p"], d["micro_r"], d["micro_f1"], d["macro_f1"],
                 t["micro_p"], t["micro_r"], t["micro_f1"], t["macro_f1"]))

    thr, top_k, d, t = best_micro
    print("\n=== selected on DEV micro-F1: threshold=%.2f top_k=%d ===" % (thr, top_k))
    for name, res in (("DEV", d), ("TEST", t)):
        print("%s  micro P/R/F1 = %.4f/%.4f/%.4f  macro-F1 = %.4f  "
              "exact-match = %.4f  predicted=%d gold=%d"
              % (name, res["micro_p"], res["micro_r"], res["micro_f1"],
                 res["macro_f1"], res["exact_match"], res["num_predicted"],
                 res["num_gold"]))
    print("\nTEST macro-F1 by train-frequency bucket:")
    for name, cnt, f1 in bucket_report(t, freq_train, num_labels):
        print("  %-12s labels=%-4d macro-F1=%.4f" % (name, cnt, f1))

    top_labels = sorted(range(num_labels), key=lambda j: -freq_train[j])[:10]
    print("\nTEST per-label F1 of the 10 most frequent labels:")
    for j in top_labels:
        print("  %-12s support=%-5d F1=%.4f"
              % (id_to_label[j], int(t["support"][j]), t["per_label_f1"][j]))


if __name__ == "__main__":
    main()
