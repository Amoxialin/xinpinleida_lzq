#!/usr/bin/env python
# coding: utf-8
"""Single-sentence / long-text / interactive prediction with the trained yft model.

Examples:
    python3 predict_one.py "珀莱雅双抗精华真好用"
    python3 predict_one.py                       # 交互模式，一行一句，空行或 Ctrl+C 退出
    python3 predict_one.py -f memo.txt --whole-file   # 把整个文件当成一篇长文本
    python3 predict_one.py -f sentences.txt      # 一行一篇
    python3 predict_one.py --top 15 --threshold 0.2 "直播间下单更便宜"
    python3 predict_one.py --no-split "很长的文本..."   # 关闭自动分段（只看前 64 字）
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 分句/分片用的分隔符（含 OCR 常见的竖线、空格、破折号等）
DELIM_RE = re.compile(u"[，。！？；、,\\.!?;:：|｜/\\\\\\s\\-—…·~～\\[\\]（）()【】{}\"'“”‘’]+")
LINE_RE = re.compile(u"[\\n\\r]+")


def build_line(text, tokenize):
    return json.dumps({
        "doc_label": [],
        "doc_token": tokenize(text),
        "doc_keyword": [],
        "doc_topic": []
    }, ensure_ascii=False)


def split_chunks(text, chunk_len, stride=None, min_len=2):
    """把长文本切成若干个长度 <= chunk_len 的片段（保留重叠）。

    训练集文本中位数只有 9~10 字，所以片段越接近训练分布，预测越准。
    min_len 用于过滤 "58"、"30元" 这类过短的 OCR 碎片，减少噪声命中。
    """
    if stride is None:
        stride = max(1, chunk_len // 2)
    chunks = []
    for line in LINE_RE.split(text.replace(u"\u3000", " ")):
        line = line.strip()
        if not line:
            continue
        for frag in DELIM_RE.split(line):
            frag = frag.strip()
            if len(frag) < min_len:
                continue
            if len(frag) <= chunk_len:
                chunks.append(frag)
                continue
            start = 0
            while True:
                win = frag[start:start + chunk_len]
                if len(win) >= min_len:
                    chunks.append(win)
                if start + chunk_len >= len(frag):
                    break
                start += stride
    return chunks


def predict_all(predictor, chunk_lists, tokenize):
    """对多篇文本（每篇是若干片段）批量预测，返回 [(probs矩阵, 片段列表), ...]

    重复片段只预测一次（长文里 "备忘录"、"14:57" 这类会大量重复）。
    """
    uniq_list, uniq_idx = [], {}
    owner_chunks = []
    for chunks in chunk_lists:
        idxs = []
        for c in chunks:
            if not tokenize(c):
                continue
            if c not in uniq_idx:
                uniq_idx[c] = len(uniq_list)
                uniq_list.append(c)
            idxs.append(uniq_idx[c])
        owner_chunks.append(idxs)

    results = [None] * len(chunk_lists)
    if not uniq_list:
        return results
    probs = predictor.predict([build_line(t, tokenize) for t in uniq_list])
    for owner, idxs in enumerate(owner_chunks):
        if not idxs:
            continue
        results[owner] = (probs[idxs], [uniq_list[i] for i in idxs])
    return results


def report(text, res, id_to_label, threshold, top_n, show_evidence,
           max_token_len, chunk_len):
    n_chars = len(text.replace("\n", "").replace(" ", ""))
    print(u"\n" + u"=" * 62)
    if res is None:
        print(u"输入: %s" % text[:40])
        print(u"  没有可识别的中文/字母数字字符，已跳过")
        return
    probs_mat, chunks = res
    agg = probs_mat.max(axis=0)
    best_chunk = probs_mat.argmax(axis=0)

    if len(chunks) > 1:
        print(u"输入: %s ..." % text[:28].replace("\n", " "))
        print(u"  全文 %d 字 → 切成 %d 个片段（每段<=%d字）后逐段预测取最大值"
              % (n_chars, len(chunks), chunk_len))
    else:
        print(u"输入: %s" % text)
        if n_chars > max_token_len:
            print(u"  注意: 超过 max_token_len=%d，模型只看了前 %d 个字"
                  % (max_token_len, max_token_len))

    order = np.argsort(-agg)
    hits = [(id_to_label[int(i)], float(agg[i]), int(best_chunk[i]))
            for i in order if agg[i] > threshold]
    if hits:
        print(u"  命中(阈值%.2f) 共%d个:" % (threshold, len(hits)))
        for name, p, ci in hits:
            if show_evidence and len(chunks) > 1:
                ev = chunks[ci].replace("\n", " ")
                ev = ev if len(ev) <= 26 else ev[:26] + u"…"
                print(u"    %-14s %.4f   ← \"%s\"" % (name, p, ev))
            else:
                print(u"    %-14s %.4f" % (name, p))
    else:
        print(u"  命中(阈值%.2f): 无" % threshold)

    print(u"  Top%d 概率:" % top_n)
    for rank, i in enumerate(order[:top_n], 1):
        mark = u"*" if agg[i] > threshold else u" "
        print(u"   %s%2d. %-14s %.4f" % (mark, rank, id_to_label[int(i)],
                                          float(agg[i])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", help=u"要预测的句子")
    parser.add_argument("--conf", default="conf/train.yft.json")
    parser.add_argument("--model", default=None, help=u"覆盖 conf 里的 eval.model_dir")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None,
                        help=u"默认使用 conf 里的 eval.threshold")
    parser.add_argument("--chunk", type=int, default=32,
                        help=u"长文本分段长度（默认32，训练集中位数9~10字）")
    parser.add_argument("--min-chunk", type=int, default=2,
                        help=u"丢弃短于该长度的碎片，OCR 噪声多时建议设 4")
    parser.add_argument("--no-split", action="store_true",
                        help=u"关闭长文本自动分段（只按 max_token_len 截断）")
    parser.add_argument("--no-evidence", action="store_true",
                        help=u"不显示命中标签对应的原文片段")
    parser.add_argument("-f", "--file", default=None, help=u"批量预测的文本文件")
    parser.add_argument("--whole-file", action="store_true",
                        help=u"配合 -f：把整个文件当成一篇长文本")
    args = parser.parse_args()

    conf_path = args.conf if os.path.isabs(args.conf) else os.path.abspath(args.conf)
    os.chdir(HERE)          # conf 里的 data/ dict_yft/ 都是相对项目根目录

    from config import Config
    from predict import Predictor
    from prepare_yft_data import tokenize

    conf = Config(config_file=conf_path)
    if args.model:
        conf.eval.model_dir = args.model
    threshold = conf.eval.threshold if args.threshold is None else args.threshold
    max_token_len = conf.feature.max_token_len

    print(u"loading model: %s ..." % conf.eval.model_dir)
    predictor = Predictor(conf)
    id_to_label = {v: k for k, v in predictor.dataset.label_map.items()}
    print(u"ready. labels=%d threshold=%.2f top=%d max_token_len=%d chunk=%d split=%s"
          % (len(id_to_label), threshold, args.top, max_token_len, args.chunk,
             not args.no_split))

    texts = []
    if args.file:
        with open(args.file, encoding="utf-8") as fin:
            content = fin.read()
        texts = [content.strip()] if args.whole_file \
            else [l.strip() for l in content.splitlines() if l.strip()]
    elif args.text:
        texts = [u" ".join(args.text)]

    if texts:
        run(predictor, texts, id_to_label, threshold, args, tokenize,
            max_token_len)
        return

    print(u"\n交互模式：输入一句话回车即可预测，直接回车或 Ctrl+C 退出。")
    print(u"（长文本请改用 -f 文件 --whole-file，交互模式一次只收一行）")
    while True:
        try:
            line = input(u"\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        run(predictor, [line], id_to_label, threshold, args, tokenize,
            max_token_len)


def run(predictor, texts, id_to_label, threshold, args, tokenize, max_token_len):
    chunk_lists = []
    for t in texts:
        toks = tokenize(t)
        if not args.no_split and len(toks) > max(16, args.chunk):
            chunk_lists.append(split_chunks(t, args.chunk,
                                            min_len=args.min_chunk))
        else:
            chunk_lists.append([t])
    results = predict_all(predictor, chunk_lists, tokenize)
    for t, res in zip(texts, results):
        report(t, res, id_to_label, threshold, args.top,
               not args.no_evidence, max_token_len, args.chunk)


if __name__ == "__main__":
    main()
