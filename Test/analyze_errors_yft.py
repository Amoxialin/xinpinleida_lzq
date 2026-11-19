#!/usr/bin/env python
# coding: utf-8
"""Error analysis for the yft multi-label model: quantify why the metrics are low.

Usage:
    python3 analyze_errors_yft.py conf/train.yft.json
"""

import json
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from config import Config
from dataset.classification_dataset import ClassificationDataset
from model.classification.textcnn import TextCNN
from tune_threshold_yft import build_loader, predict_probs, metrics


def load_json(path):
    rows = []
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def content_of(row):
    return "".join(row["doc_token"])


def main():
    conf = Config(config_file=sys.argv[1])
    thr, top_k = conf.eval.threshold, conf.eval.top_k

    test_dataset, test_loader = build_loader(conf, conf.data.test_json_files)
    label_map = test_dataset.label_map
    num_labels = len(label_map)
    id_to_label = {v: k for k, v in label_map.items()}

    model = TextCNN(ClassificationDataset(conf, []), conf)
    if conf.device.startswith("cuda"):
        model = model.cuda(conf.device)
    ckpt = torch.load(conf.eval.model_dir, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    probs, golds = predict_probs(conf, model, test_loader)
    train_rows = load_json(conf.data.train_json_files[0])
    test_rows = load_json(conf.data.test_json_files[0])
    assert len(test_rows) == len(probs), (len(test_rows), len(probs))

    print("=" * 66)
    print("baseline: thr=%.2f top_k=%d  test microF1=%.4f macroF1=%.4f exact=%.4f"
          % (thr, top_k,
             metrics(probs, golds, thr, top_k, num_labels)["micro_f1"],
             metrics(probs, golds, thr, top_k, num_labels)["macro_f1"],
             metrics(probs, golds, thr, top_k, num_labels)["exact_match"]))

    # ---------- 1. 标注冲突：同一文本在训练集里被打了不同标签 ----------
    train_by_content = defaultdict(set)
    for r in train_rows:
        train_by_content[content_of(r)].add(tuple(sorted(r["doc_label"])))
    conflict_contents = {c for c, s in train_by_content.items() if len(s) > 1}

    seen = set()
    conflict_idx, dup_idx, clean_idx = [], [], []
    for i, r in enumerate(test_rows):
        c = content_of(r)
        if c in conflict_contents:
            conflict_idx.append(i)
        elif c in seen:
            dup_idx.append(i)
        else:
            clean_idx.append(i)
        seen.add(c)

    def subset_f1(idxs):
        if not idxs:
            return 0.0, 0.0, 0
        res = metrics(probs[idxs], [golds[i] for i in idxs], thr, top_k,
                      num_labels)
        return res["micro_f1"], res["exact_match"], len(idxs)

    print("\n[1] 标注冲突（同一文本在训练集里出现多种标签组合）")
    print("    训练集中冲突文本组数: %d" % len(conflict_contents))
    for name, idxs in (("冲突样本", conflict_idx), ("重复样本", dup_idx),
                       ("其余样本", clean_idx)):
        f1, ex, n = subset_f1(idxs)
        print("    测试集 %-6s n=%-5d microF1=%.4f exact=%.4f"
              % (name, n, f1, ex))

    # ---------- 2. 近义/包含关系标签造成的混淆 ----------
    order = np.argsort(-probs, axis=1)
    confusion = Counter()
    contain_fp = 0
    total_fp = 0
    for i, prob in enumerate(probs):
        idx = order[i][:top_k]
        idx = idx[prob[idx] > thr]
        pred = set(int(x) for x in idx)
        gold = set(int(x) for x in golds[i])
        for g in gold - pred:
            for p in pred - gold:
                confusion[(g, p)] += 1
        total_fp += len(pred - gold)
        for p in pred - gold:
            pn = id_to_label[p]
            if any(pn in id_to_label[g] or id_to_label[g] in pn for g in gold):
                contain_fp += 1

    print("\n[2] 近义标签混淆（真值→误判，Top15）")
    for (g, p), c in confusion.most_common(15):
        rel = "包含关系" if (id_to_label[g] in id_to_label[p]
                             or id_to_label[p] in id_to_label[g]) else ""
        print("    %-12s → %-12s %4d 次 %s"
              % (id_to_label[g], id_to_label[p], c, rel))
    print("    误判(FP)总数=%d，其中与真值标签存在名称包含关系的=%d (%.1f%%)"
          % (total_fp, contain_fp, 100.0 * contain_fp / max(1, total_fp)))

    # ---------- 3. 阈值/top_k 造成的漏报 ----------
    fn_ranges = Counter()
    total_fn = 0
    for i, prob in enumerate(probs):
        idx = order[i][:top_k]
        idx = idx[prob[idx] > thr]
        pred = set(int(x) for x in idx)
        gold = set(int(x) for x in golds[i])
        for g in gold - pred:
            total_fn += 1
            p = float(prob[g])
            if p >= 0.9 * thr:
                fn_ranges["0.9*thr~thr (差一点)"] += 1
            elif p >= 0.5 * thr:
                fn_ranges["0.5*thr~0.9*thr"] += 1
            elif p >= 0.1:
                fn_ranges["0.1~0.5*thr"] += 1
            else:
                fn_ranges["<0.1 (完全没学到)"] += 1
        # top_k 截断：真值标签概率超过阈值但排在 top_k 之外
    cut = 0
    for i, prob in enumerate(probs):
        gold = set(int(x) for x in golds[i])
        above = set(int(x) for x in np.where(prob > thr)[0])
        ranked = set(int(x) for x in order[i][:top_k])
        cut += len(gold & above - ranked)

    print("\n[3] 漏报(FN=%d)的概率分布" % total_fn)
    for k in ("0.9*thr~thr (差一点)", "0.5*thr~0.9*thr", "0.1~0.5*thr",
              "<0.1 (完全没学到)"):
        print("    %-22s %5d (%.1f%%)"
              % (k, fn_ranges[k], 100.0 * fn_ranges[k] / max(1, total_fn)))
    print("    其中因 top_k=%d 截断而漏掉的(概率已超阈值但排名靠后): %d"
          % (top_k, cut))

    print("\n[4] 放宽 top_k 的影响（阈值不变 thr=%.2f）" % thr)
    for k in (top_k, 5, 10, 478):
        res = metrics(probs, golds, thr, k, num_labels)
        print("    top_k=%-4d microP=%.4f microR=%.4f microF1=%.4f macroF1=%.4f"
              % (k, res["micro_p"], res["micro_r"], res["micro_f1"],
                 res["macro_f1"]))

    # ---------- 5. 长尾：训练频次 vs F1 ----------
    freq_train = np.zeros(num_labels)
    for r in train_rows:
        for lb in r["doc_label"]:
            if lb in label_map:
                freq_train[label_map[lb]] += 1
    res = metrics(probs, golds, thr, top_k, num_labels)
    supported = res["support"] > 0
    f1s, freqs = res["per_label_f1"][supported], freq_train[supported]
    print("\n[5] 长尾效应")
    for name, lo, hi in (("训练<5条", 0, 5), ("5~19条", 5, 20),
                         ("20~99条", 20, 100), (">=100条", 100, 10 ** 9)):
        m = (freqs >= lo) & (freqs < hi)
        if m.sum():
            print("    %-10s 标签数=%-4d 平均F1=%.4f  F1=0的标签数=%d"
                  % (name, m.sum(), f1s[m].mean(), int((f1s[m] == 0).sum())))
    print("    训练频次与F1的相关系数: %.3f" % np.corrcoef(freqs, f1s)[0, 1])

    # ---------- 6. 输入截断 ----------
    print("\n[6] 输入截断 (max_token_len=%d)" % conf.feature.max_token_len)
    for name, rows in (("train", train_rows), ("test", test_rows)):
        lens = np.array([len(content_of(r)) for r in rows])
        over = int((lens > conf.feature.max_token_len).sum())
        print("    %-5s 长度 p50=%d p99=%d max=%d  超长被截断=%d (%.2f%%)"
              % (name, int(np.percentile(lens, 50)),
                 int(np.percentile(lens, 99)), lens.max(), over,
                 100.0 * over / len(lens)))


if __name__ == "__main__":
    main()
