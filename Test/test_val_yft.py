#!/usr/bin/env python
# coding: utf-8
"""在 test-val.csv 上评测已训练的 yft 模型（TextRNN BiGRU+AVG）。

流程：
  1. 读取 CSV（7 列：序号 / video_highlight_content / audio_asr_content /
     content / cover_ocr_content / title / 标签去重）
  2. 把 5 个文本列合并成一篇帖子文本，按方案做滑窗切分
     （换行 -> 标点分句 -> 超过 chunk 字的片段用步长 chunk/2 滑窗）
  3. 逐片段推理，每个标签取所有片段的最大概率，超过阈值即判为命中
  4. 与「标签去重」列的金标签对比，输出：
       - 全部数据的整体指标：micro P/R/F1、macro P/R/F1、样本级 P/R/F1、
         子集准确率(exact match)、Hamming 准确率
       - test-val_predicted.csv：原始 CSV 的所有列原样保留，行尾追加
         「预测标签 / 预测标签(带概率) / 多预测(FP) / 漏预测(FN) /
           预测正确数(TP) / 片段数」，方便肉眼逐行对比
       - 逐标签 P/R/F1（按 support 排序，写入单独 CSV）

产物目录：每次运行都在 logs/ 下新建一个「<模型>_<时间>」目录，互不覆盖，例如
    logs/TextRNN_best_20261119_143022/
        ├── run.log                 本次完整控制台输出（含总指标表）
        ├── run_meta.json           命令行、全部参数、数据规模、总指标
        ├── test-val_predicted.csv  逐行预测结果
        ├── test_val_per_label.csv  逐标签 P/R/F1
        ├── framework.log           框架自身日志（原 logs/train_yft_run.log）
        └── probs_agg_gold.npz      仅 --save-probs 时生成
模型标识取 conf.eval.model_dir 的目录名（如 checkpoint_dir_yft/TextRNN_best
-> TextRNN_best），时间为运行开始时刻 %Y%m%d_%H%M%S。

用法（在项目根目录下运行）：
    python test_val_yft.py                          # 默认 data/test-val.csv
    python test_val_yft.py --threshold 0.4          # 换阈值
    python test_val_yft.py --show 30                # 控制台多预览几行
    python test_val_yft.py --run-dir logs/my_test   # 指定产物目录（复用，不自动命名）
    python test_val_yft.py --out /tmp/result.csv    # 输出到目录外（带路径/绝对路径时尊重原样）
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from predict_one import build_line, split_chunks  # noqa: E402  复用滑窗切分

# 金标签分隔符：、 ； ; | ， ,   （注意不能按 / 切，标签里有"正品/假货"）
LABEL_SEP_RE = re.compile(u"[、;；|,，]+")

DEFAULT_TEXT_COLS = ("video_highlight_content,audio_asr_content,"
                     "content,cover_ocr_content,title")

LOG_ROOT = os.path.join(HERE, "logs")


# ---------------------------------------------------------------- 运行目录

class Tee(object):
    """把 stdout 同时写到控制台和 run.log，让每次运行的输出都能留档。"""

    def __init__(self, stream, path):
        self.stream = stream
        self.fout = open(path, "a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.fout.write(s)
        self.fout.flush()

    def flush(self):
        self.stream.flush()
        self.fout.flush()

    def close(self):
        try:
            self.fout.close()
        except Exception:
            pass


def sanitize_tag(text):
    """把模型名里的路径分隔符等换成下划线，保证能当目录名。"""
    return re.sub(u"[^0-9A-Za-z_.-]+", u"_", (text or u"model").strip()) or u"model"


def model_tag(conf):
    """用 checkpoint 目录名当模型标识：checkpoint_dir_yft/TextRNN_best -> TextRNN_best"""
    name = getattr(conf, "model_name", None) or u"model"
    ckpt = os.path.basename(str(conf.eval.model_dir).rstrip(u"/\\"))
    if not ckpt or ckpt == name:
        return name
    return ckpt if ckpt.startswith(name) else u"%s_%s" % (name, ckpt)


def make_run_dir(tag, override=None):
    """建本次运行的产物目录。

    override 为空时在 logs/ 下自动命名 <模型>_<时间>，同名则追加 -2/-3；
    指定了 override 则复用该目录：只给名字时默认也放在 logs/ 下，
    带路径或绝对路径时按原样解析。
    """
    if override:
        if os.path.isabs(override):
            path = override
        elif os.path.dirname(override):
            path = os.path.join(HERE, override)
        else:
            path = os.path.join(LOG_ROOT, override)
        os.makedirs(path, exist_ok=True)
        return path
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(LOG_ROOT, u"%s_%s" % (sanitize_tag(tag), stamp))
    path, i = base, 1
    while os.path.exists(path):
        i += 1
        path = u"%s-%d" % (base, i)
    os.makedirs(path)
    return path


def resolve_out(value, run_dir, default_name):
    """输出路径解析规则：

    - 没指定        -> <run_dir>/<默认名>
    - 只给文件名    -> <run_dir>/<文件名>（跟着运行目录走）
    - 带目录或绝对路径 -> 原样使用（明确要写到目录外）
    """
    if not value:
        return os.path.join(run_dir, default_name)
    if os.path.isabs(value) or os.path.dirname(value):
        return value
    return os.path.join(run_dir, value)


# ---------------------------------------------------------------- CSV 读取

def raise_field_limit():
    """单元格里可能有超长 OCR 文本，放宽 csv 默认 128KB 限制。"""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit // 10)


def read_rows(path, encoding=None):
    """读 CSV 返回 (header, rows)。自动处理 BOM，编码失败时回退 gb18030。"""
    raise_field_limit()
    for enc in ([encoding] if encoding else ["utf-8-sig", "utf-8", "gb18030"]):
        try:
            with open(path, "r", encoding=enc, newline="") as fin:
                reader = csv.reader(fin)
                rows = [r for r in reader]
            print(u"读取成功: %s (编码=%s, 总行数=%d)" % (path, enc, len(rows)))
            break
        except UnicodeDecodeError:
            continue
    else:
        sys.exit(u"[错误] 无法识别文件编码: %s" % path)

    if not rows:
        sys.exit(u"[错误] CSV 是空的")
    header = [h.strip().lstrip(u"\ufeff") for h in rows[0]]
    data = [r for r in rows[1:] if any((c or "").strip() for c in r)]
    return header, data


def col_index(header, name):
    for i, h in enumerate(header):
        if h == name:
            return i
    sys.exit(u"[错误] 找不到列「%s」，实际表头: %s" % (name, u" | ".join(header)))


def cell(row, i):
    return str(row[i]).strip() if i < len(row) and row[i] is not None else ""


def parse_gold_labels(text):
    """'品牌提及、双抗精华、软广' -> ['品牌提及', '双抗精华', '软广']"""
    parts = LABEL_SEP_RE.split(text or "")
    labels = []
    for p in parts:
        p = p.strip().strip(u'"\'“”‘’').strip()
        p = p.replace(u"（", "(").replace(u"）", ")")   # 全角括号归一化
        if p and p not in labels:
            labels.append(p)
    return labels


# ---------------------------------------------------------------- 指标

def prf(tp, fp, fn):
    p = tp / float(tp + fp) if tp + fp else 0.0
    r = tp / float(tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def compute_metrics(golds, preds, vocab_size):
    """golds/preds: List[set]。返回 (整体指标 dict, 逐标签 dict)。"""
    tp = fp = fn = 0
    sample_p, sample_r, sample_f = [], [], []
    exact = 0
    hamming = []
    per_label = {}   # label -> [tp, fp, fn]

    for g, p in zip(golds, preds):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        if g == p:
            exact += 1
        if g or p:
            sp, sr, sf = prf(len(g & p), len(p - g), len(g - p))
            sample_p.append(sp), sample_r.append(sr), sample_f.append(sf)
        hamming.append(1.0 - len(g ^ p) / float(vocab_size))
        for lab in g | p:
            st = per_label.setdefault(lab, [0, 0, 0])
            if lab in g and lab in p:
                st[0] += 1
            elif lab in p:
                st[1] += 1
            else:
                st[2] += 1

    micro = prf(tp, fp, fn)
    n = max(len(golds), 1)
    label_stats = {}
    for lab, (ltp, lfp, lfn) in per_label.items():
        p, r, f = prf(ltp, lfp, lfn)
        label_stats[lab] = {"support": ltp + lfn, "tp": ltp, "fp": lfp,
                            "fn": lfn, "P": p, "R": r, "F1": f}
    macro_p = np.mean([v["P"] for v in label_stats.values()]) if label_stats else 0.
    macro_r = np.mean([v["R"] for v in label_stats.values()]) if label_stats else 0.
    macro_f = np.mean([v["F1"] for v in label_stats.values()]) if label_stats else 0.

    overall = {
        "rows": len(golds),
        "micro_P": micro[0], "micro_R": micro[1], "micro_F1": micro[2],
        "macro_P": macro_p, "macro_R": macro_r, "macro_F1": macro_f,
        "sample_P": np.mean(sample_p) if sample_p else 0.,
        "sample_R": np.mean(sample_r) if sample_r else 0.,
        "sample_F1": np.mean(sample_f) if sample_f else 0.,
        "exact_match": exact / float(n),
        "hamming_acc": np.mean(hamming) if hamming else 0.,
        "TP": tp, "FP": fp, "FN": fn,
        "gold_labels": sum(len(g) for g in golds),
        "pred_labels": sum(len(p) for p in preds),
    }
    return overall, label_stats


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description=u"real_1000.csv 模型评测")
    ap.add_argument("--csv", default="data/real_1000.csv", help=u"输入 CSV")
    ap.add_argument("--conf", default="conf/train.yft.json")
    ap.add_argument("--model", default=None, help=u"覆盖 conf.eval.model_dir")
    ap.add_argument("--id-col", default=u"序号")
    ap.add_argument("--text-cols", default=DEFAULT_TEXT_COLS,
                    help=u"参与预测的文本列，逗号分隔")
    ap.add_argument("--label-col", default=u"标签去重")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help=u"判定阈值（默认0.5，已验证的最优操作点）")
    ap.add_argument("--chunk", type=int, default=32, help=u"滑窗长度")
    ap.add_argument("--min-chunk", type=int, default=2,
                    help=u"丢弃短于该长度的碎片")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--no-split", action="store_true",
                    help=u"关闭滑窗切分（整篇截断到 max_token_len）")
    ap.add_argument("--min-cn-ratio", type=float, default=0.3,
                    help=u"片段汉字占比低于该值则丢弃（过滤OCR价格/时间戳等脏碎片；"
                         u"默认0.3，0=不过滤）")
    ap.add_argument("--min-support", type=float, default=-0.02,
                    help=u"标签至少要在几个片段上过阈值才算命中；"
                         u">=1 是片段个数，<1 是占片段总数的比例"
                         u"（默认-0.02=至少2%%的片段支持）")
    ap.add_argument("--max-labels", type=int, default=15,
                    help=u"每行最多输出几个标签，按概率取前 N（默认15，0=不限）")
    ap.add_argument("--save-probs", nargs="?", const="", default=None,
                    help=u"把逐行聚合概率矩阵和金标签存成 npz（写进本次运行目录），"
                         u"便于离线扫阈值；只写 --save-probs 用默认名 probs_agg_gold.npz，"
                         u"不加该参数则不保存")
    ap.add_argument("--out", default=None,
                    help=u"逐行预测结果 CSV：原始列+预测列（utf-8-sig，可直接用 Excel 打开）；"
                         u"默认写入本次运行目录的 test-val_predicted.csv")
    ap.add_argument("--per-label-out", default=None,
                    help=u"逐标签指标 CSV；默认写入本次运行目录的 test_val_per_label.csv")
    ap.add_argument("--show", type=int, default=0,
                    help=u"控制台预览前 N 行的 金标签 vs 预测标签（默认0=不预览，只输出总指标）")
    ap.add_argument("--limit", type=int, default=0, help=u"只跑前 N 行，0=全部")
    ap.add_argument("--run-dir", default=None,
                    help=u"产物目录；默认 logs/<模型>_<时间> 自动新建，指定后复用该目录")
    args = ap.parse_args()

    t_start = time.time()
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(HERE, args.csv)
    conf_path = args.conf if os.path.isabs(args.conf) \
        else os.path.join(HERE, args.conf)
    os.chdir(HERE)   # conf 里 data/ dict_yft/ 等是相对项目根目录

    # ---- 0) 建本次运行目录：logs/<模型>_<时间>，所有产物都落在里面
    from config import Config

    conf = Config(config_file=conf_path)
    if args.model:
        conf.eval.model_dir = args.model
    run_dir = make_run_dir(model_tag(conf), args.run_dir)
    tee = Tee(sys.stdout, os.path.join(run_dir, "run.log"))
    sys.stdout = tee
    # 框架自身的 logging 也跟着进运行目录（原来固定写 logs/train_yft_run.log）
    if getattr(conf, "log", None) is not None:
        conf.log.logger_file = os.path.join(run_dir, "framework.log")
    print(u"本次运行目录: %s" % run_dir)
    print(u"开始时间: %s  命令行: %s" % (start_time, u" ".join(sys.argv)))

    # ---- 1) 读数据
    header, data = read_rows(csv_path)
    if args.limit > 0:
        data = data[:args.limit]
    id_i = col_index(header, args.id_col)
    label_i = col_index(header, args.label_col)
    text_is = [col_index(header, c.strip())
               for c in args.text_cols.split(",") if c.strip()]
    print(u"数据行=%d  ID列=%s  文本列=%s  金标签列=%s"
          % (len(data), args.id_col, args.text_cols, args.label_col))

    # ---- 2) 加载模型（conf 已在第 0 步加载，复用同一个对象）
    from predict import Predictor
    from prepare_yft_data import tokenize

    threshold = conf.eval.threshold if args.threshold is None else args.threshold
    max_token_len = conf.feature.max_token_len
    print(u"加载模型: %s (%s)  阈值=%.2f  滑窗=%d"
          % (conf.eval.model_dir, conf.model_name, threshold, args.chunk))
    predictor = Predictor(conf)
    id_to_label = {v: k for k, v in predictor.dataset.label_map.items()}
    vocab = set(id_to_label.values())
    vocab_size = len(vocab)
    print(u"模型标签数=%d" % vocab_size)

    # ---- 3) 切片段（全局去重，重复片段只推理一次）
    uniq, uniq_idx = [], {}
    row_chunk_ids, row_texts, row_golds = [], [], []
    unknown_labels = {}
    for r in data:
        parts = [cell(r, i) for i in text_is]
        text = u"\n".join(p for p in parts if p)
        row_texts.append((cell(r, id_i), text))

        toks = tokenize(text)
        if not toks:
            chunks = []
        elif args.no_split or len(toks) <= max(16, args.chunk):
            chunks = [text]
        else:
            chunks = split_chunks(text, args.chunk, min_len=args.min_chunk)
        ids = []
        for c in chunks:
            if not tokenize(c):
                continue
            if args.min_cn_ratio > 0:
                cn = sum(1 for ch in c if u'\u4e00' <= ch <= u'\u9fff')
                if cn / float(len(c)) < args.min_cn_ratio:
                    continue    # 脏碎片（数字/价格/URL等），不进模型
            if c not in uniq_idx:
                uniq_idx[c] = len(uniq)
                uniq.append(c)
            ids.append(uniq_idx[c])
        row_chunk_ids.append(ids)

        gold = parse_gold_labels(cell(r, label_i))
        for lab in gold:
            if lab not in vocab:
                unknown_labels[lab] = unknown_labels.get(lab, 0) + 1
        row_golds.append(set(l for l in gold if l in vocab))

    n_chunks = sum(len(x) for x in row_chunk_ids)
    print(u"%d 行 -> %d 个片段，去重后 %d 个待推理；无有效文本行=%d"
          % (len(data), n_chunks, len(uniq),
             sum(1 for x in row_chunk_ids if not x)))
    if unknown_labels:
        print(u"[注意] 金标签中有 %d 种不在模型标签表内（共 %d 次），已忽略: %s"
              % (len(unknown_labels), sum(unknown_labels.values()),
                 u"、".join(sorted(unknown_labels, key=unknown_labels.get,
                                   reverse=True)[:15])))

    # ---- 4) 分批推理
    t0 = time.time()
    probs = None
    for start in range(0, len(uniq), args.batch_size):
        part = uniq[start:start + args.batch_size]
        p = predictor.predict([build_line(t, tokenize) for t in part])
        if probs is None:
            probs = np.zeros((len(uniq), p.shape[1]), dtype=np.float32)
        probs[start:start + len(part)] = p
        print(u"  推理 %d/%d 片段 (%.1fs)"
              % (min(start + args.batch_size, len(uniq)), len(uniq),
                 time.time() - t0))

    # ---- 5) 聚合 + 判定（max概率 + 片段支持数过滤 + 每行标签上限）
    n_labels = probs.shape[1] if probs is not None else 0
    agg_all = np.zeros((len(data), n_labels), dtype=np.float32)
    preds, pred_probs = [], []
    for k, ids in enumerate(row_chunk_ids):
        if not ids:
            preds.append(set())
            pred_probs.append({})
            continue
        sub = probs[ids]
        agg = sub.max(axis=0)                 # 每标签取所有片段最大概率
        agg_all[k] = agg
        support = (sub > threshold).sum(axis=0)   # 过阈值的片段数
        need = args.min_support if args.min_support >= 1 \
            else max(1, int(np.ceil(-args.min_support * len(ids))))
        ok = np.where((agg > threshold) & (support >= need))[0]
        if args.max_labels > 0 and len(ok) > args.max_labels:
            ok = ok[np.argsort(-agg[ok])[:args.max_labels]]
        hit = {id_to_label[int(i)]: float(agg[i]) for i in ok}
        preds.append(set(hit.keys()))
        pred_probs.append(hit)

    probs_path = None
    if args.save_probs is not None:
        sp = resolve_out(args.save_probs, run_dir, "probs_agg_gold.npz")
        gold_mat = np.zeros((len(data), n_labels), dtype=bool)
        for k, g in enumerate(row_golds):
            for lab in g:
                gold_mat[k, predictor.dataset.label_map[lab]] = True
        np.savez_compressed(sp, agg=agg_all, gold=gold_mat)
        probs_path = sp
        print(u"概率矩阵已保存: %s（可离线扫阈值）" % sp)

    # ---- 6) 指标：输出全部数据的总指标汇总表
    overall, label_stats = compute_metrics(row_golds, preds, vocab_size)
    topk_desc = (u"top%d" % args.max_labels) if args.max_labels > 0 else u"不限"
    print(u"\n" + u"=" * 62)
    print(u"全部 %d 条数据的总指标（%s，阈值=%.2f，每行标签上限=%s）"
          % (overall["rows"], conf.model_name, threshold, topk_desc))
    print(u"-" * 62)
    print(u"%-38s%s" % (u"指标", u"值"))
    summary_rows = [
        (u"micro Precision", u"%.4f" % overall["micro_P"]),
        (u"micro Recall", u"%.4f" % overall["micro_R"]),
        (u"micro F1", u"%.4f" % overall["micro_F1"]),
        (u"sample级 P / R / F1（逐帖平均）", u"%.3f / %.3f / %.3f"
         % (overall["sample_P"], overall["sample_R"], overall["sample_F1"])),
        (u"macro P / R / F1", u"%.3f / %.3f / %.3f"
         % (overall["macro_P"], overall["macro_R"], overall["macro_F1"])),
        (u"子集准确率（exact match，整帖全对）", u"%.4f" % overall["exact_match"]),
        (u"Hamming 准确率", u"%.4f" % overall["hamming_acc"]),
        (u"TP / FP / FN", u"%d / %d / %d"
         % (overall["TP"], overall["FP"], overall["FN"])),
        (u"金标签总数 / 预测标签总数", u"%d / %d"
         % (overall["gold_labels"], overall["pred_labels"])),
    ]
    for name, val in summary_rows:
        print(u"%-40s%s" % (name, val))
    print(u"=" * 62)

    # ---- 7) 控制台预览
    if args.show > 0:
        print(u"\n前 %d 行 预测 vs 金标签:" % min(args.show, len(data)))
        for k in range(min(args.show, len(data))):
            pid, text = row_texts[k]
            g, p = row_golds[k], preds[k]
            print(u"  [%s] 文本%d字 片段%d个" % (pid, len(text), len(row_chunk_ids[k])))
            print(u"    金标签: %s" % (u"、".join(sorted(g)) or u"(无)"))
            print(u"    预测  : %s" % (u"、".join(
                u"%s(%.2f)" % (l, pred_probs[k][l])
                for l in sorted(p, key=lambda x: -pred_probs[k][x])) or u"(无)"))
            if p - g:
                print(u"    多预测: %s" % u"、".join(sorted(p - g)))
            if g - p:
                print(u"    漏预测: %s" % u"、".join(sorted(g - p)))

    # ---- 8) 写逐行结果 CSV：原始输入所有列原样保留 + 追加预测列，方便肉眼对比
    out_path = resolve_out(args.out, run_dir, "test-val_predicted.csv")
    extra_cols = [u"预测标签", u"预测标签(带概率)", u"多预测(FP)",
                  u"漏预测(FN)", u"预测正确数(TP)", u"片段数"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header + extra_cols)
        for k in range(len(data)):
            pid, text = row_texts[k]
            g, p = row_golds[k], preds[k]
            orig = list(data[k]) + [u""] * (len(header) - len(data[k]))
            w.writerow(orig + [
                u"、".join(sorted(p)),
                u"、".join(u"%s:%.3f" % (l, pred_probs[k][l])
                           for l in sorted(p, key=lambda x: -pred_probs[k][x])),
                u"、".join(sorted(p - g)),
                u"、".join(sorted(g - p)),
                len(g & p),
                len(row_chunk_ids[k]),
            ])
    print(u"\n逐行结果已写出: %s（原始 %d 列 + %d 个预测列）"
          % (out_path, len(header), len(extra_cols)))

    # ---- 9) 写逐标签指标 CSV（按 support 降序）
    pl_path = resolve_out(args.per_label_out, run_dir, "test_val_per_label.csv")
    with open(pl_path, "w", encoding="utf-8-sig", newline="") as fout:
        w = csv.writer(fout)
        w.writerow([u"标签", u"support(金标签出现次数)", u"precision",
                    u"recall", u"F1", u"TP", u"FP", u"FN"])
        for lab, st in sorted(label_stats.items(),
                              key=lambda x: -x[1]["support"]):
            w.writerow([lab, st["support"], u"%.4f" % st["P"],
                        u"%.4f" % st["R"], u"%.4f" % st["F1"],
                        st["tp"], st["fp"], st["fn"]])
    print(u"逐标签指标已写出: %s (%d 个标签)" % (pl_path, len(label_stats)))

    # ---- 10) 写运行元信息（命令行、参数、数据规模、总指标），便于事后追溯
    meta = {
        u"开始时间": start_time,
        u"耗时秒": round(time.time() - t_start, 1),
        u"运行目录": run_dir,
        u"命令行": u" ".join(sys.argv),
        u"参数": vars(args),
        u"输入CSV": csv_path,
        u"模型权重": conf.eval.model_dir,
        u"模型结构": conf.model_name,
        u"数据行数": len(data),
        u"阈值": threshold,
        u"滑窗长度": args.chunk,
        u"片段总数": n_chunks,
        u"去重后推理片段数": len(uniq),
        u"模型标签数": vocab_size,
        u"词表外金标签": unknown_labels,
        u"总指标": overall,
        u"产物文件": {
            u"逐行结果": out_path,
            u"逐标签指标": pl_path,
            u"运行日志": os.path.join(run_dir, "run.log"),
            u"概率矩阵": probs_path,
        },
    }
    meta_path = os.path.join(run_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fout:
        json.dump(meta, fout, ensure_ascii=False, indent=2, default=float)

    print(u"\n本次产物目录: %s" % run_dir)
    for name in sorted(os.listdir(run_dir)):
        print(u"  %-26s %9.1f KB"
              % (name, os.path.getsize(os.path.join(run_dir, name)) / 1024.0))
    print(u"运行元信息已写出: %s" % meta_path)
    print(u"总耗时 %.1fs" % (time.time() - t_start))

    sys.stdout = tee.stream
    tee.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # 出错时把 traceback 也追加进本次运行目录的 run.log，事后能查是哪次跑挂了
        if isinstance(sys.stdout, Tee):
            import traceback
            tee_obj = sys.stdout
            tee_obj.fout.write(traceback.format_exc())
            sys.stdout = tee_obj.stream
            tee_obj.close()
        raise
