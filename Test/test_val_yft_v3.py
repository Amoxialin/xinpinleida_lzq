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
import postproc_opts as PP  # noqa: E402  《生产后处理诊断与优化方案.md》各 Opt 实现

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
    ap.add_argument("--support-cap", type=int, default=0,
                    help=u"比例式 need 的上限，0=不封顶（与旧脚本完全一致）。"
                         u"长帖片段数可达数百，need=ceil(0.02n) 会把只在少数片段出现的"
                         u"话题整体压掉，导致长帖召回崩塌；封顶后短帖不受影响、长帖恢复"
                         u"召回。推荐 2~3")
    ap.add_argument("--agg", type=str, default="max",
                    choices=["max", "mean", "topk", "noisyor"],
                    help=u"片段概率聚合方式：max=取最大（旧行为）；mean=取均值（长帖稀释"
                         u"严重）；topk=取概率最高的 k 个片段求均值；noisyor=1-Π(1-p)"
                         u"（文档 Opt-4，比 max 抗单片段噪声）")
    ap.add_argument("--agg-topk", type=int, default=3,
                    help=u"--agg topk 时的 k 值（默认3）")
    ap.add_argument("--save-chunk-probs", nargs="?", const="", default=None,
                    help=u"保存片段级概率矩阵(npz)，之后调后处理规则可离线复算、无需重跑"
                         u"推理；只写 --save-chunk-probs 用默认名 probs_chunk.npz")
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

    # ============ 以下为《生产后处理诊断与优化方案.md》各 Opt 的开关 ============
    # 默认值全部等于旧脚本行为，便于做"逐项开启"的增量归因实验。
    g = ap.add_argument_group(u"文档方案开关（默认全关=旧行为）")
    g.add_argument("--filter-mode", default="cnratio",
                   choices=["cnratio", "whitelist", "none"],
                   help=u"Opt-1 脏片段过滤：cnratio=现有汉字占比一刀切；whitelist=文档"
                        u"白名单（汉字达标 或 命中品牌/价格/规格/序号）；none=不过滤")
    g.add_argument("--train-xlsx", default="data/new_datasets.xlsx",
                   help=u"训练数据 xlsx，供 Opt-1 品牌词典与 Opt-3 检索兜底使用")
    g.add_argument("--protect-decimal", action="store_true",
                   help=u"Opt-8 分句前保护数字小数点，避免 7.5ml / 68.5元 被切碎")
    g.add_argument("--retrieval-fallback", action="store_true",
                   help=u"Opt-3 检索兜底：片段精确命中训练文本则直接抄其标签，与模型"
                        u"预测取并集。保证 macro-F1 有下界")
    g.add_argument("--support-dedup", action="store_true",
                   help=u"Opt-5 支持率计票前按片段文本去重（同一句OCR重复几十次不再刷票）")
    g.add_argument("--support-votes", type=int, default=0,
                   help=u"Opt-5 绝对票数制：>0 时弃用百分比，改为长尾标签需 k_low 票、"
                        u"高频标签需 k_high 票（配合 --support-freq-split/--support-k-high）")
    g.add_argument("--support-freq-split", type=int, default=1000,
                   help=u"Opt-5 高/低频标签的训练频次分界（默认1000）")
    g.add_argument("--support-k-high", type=int, default=3,
                   help=u"Opt-5 高频标签所需票数（默认3）")
    g.add_argument("--label-whitelist", action="store_true",
                   help=u"Opt-7 只输出生产活跃标签，屏蔽僵尸标签（训练有、生产从未出现）")
    g.add_argument("--label-whitelist-source", default="dev",
                   choices=["dev", "all"],
                   help=u"Opt-7 白名单来源：dev=从 dev 划分金标签统计（无泄漏，但 dev "
                        u"只有200帖、标签覆盖不全）；all=从全量1049帖统计（模拟生产中"
                        u"业务确认的322个在用标签，但用 test 标签分布过滤 test 预测，"
                        u"属信息泄漏，仅作方案上限参考，不可作为可落地成绩）")
    g.add_argument("--norm-long-doc", action="store_true",
                   help=u"Opt-4 聚合分数除以 log(1+片段数)，抑制长帖天然更易越阈")
    g.add_argument("--per-label-threshold", action="store_true",
                   help=u"Opt-4 在 dev 集上为每个标签单独校准阈值，替代全局单一阈值")
    g.add_argument("--calib-min-support", type=int, default=5,
                   help=u"Opt-4 逐标签校准的最小 dev 金标签数，低于此值回退全局阈值")
    g.add_argument("--split", default="all", choices=["all", "dev", "test"],
                   help=u"Opt-9 评测集划分：all=全部1049帖（调参与报分同数据，有过拟合"
                        u"风险）；dev=前200帖只用于调参；test=其余849帖只用于报分")
    g.add_argument("--n-dev", type=int, default=200, help=u"Opt-9 dev 集帖数（默认200）")
    g.add_argument("--split-seed", type=int, default=42, help=u"Opt-9 划分随机种子")
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
    # Opt-1：品牌词典只在 whitelist 模式下才需要构建（要解析训练 xlsx，有成本）
    brand = None
    if args.filter_mode == "whitelist":
        t0 = time.time()
        xlsx = args.train_xlsx if os.path.isabs(args.train_xlsx) \
            else os.path.join(HERE, args.train_xlsx)
        brand = PP.build_brand_dict(xlsx)
        print(u"[Opt-1] 白名单脏过滤：从训练集「品牌提及」自动构建品牌词典 %d 个，"
              u"耗时 %.1fs（样例: %s）"
              % (len(brand), time.time() - t0,
                 u"、".join(sorted(brand)[:8])))

    uniq, uniq_idx = [], {}
    row_chunk_ids, row_texts, row_golds = [], [], []
    row_raw_n, row_dropped = [], []       # 诊断用：切分数 / 被过滤数
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
            # Opt-8：分句前把 7.5 / 68.5 这类小数点占位保护，分句后还原
            if args.protect_decimal:
                chunks = [PP.restore_decimal(c) for c in
                          split_chunks(PP.protect_decimal(text), args.chunk,
                                       min_len=args.min_chunk)]
            else:
                chunks = split_chunks(text, args.chunk, min_len=args.min_chunk)
        row_raw_n.append(len(chunks))
        ids = []
        n_drop = 0
        for c in chunks:
            if not tokenize(c):
                n_drop += 1
                continue
            # Opt-1：cnratio=旧规则；whitelist=文档白名单；none=不过滤
            if args.filter_mode == "cnratio":
                if args.min_cn_ratio > 0:
                    cn = sum(1 for ch in c if u'\u4e00' <= ch <= u'\u9fff')
                    if cn / float(len(c)) < args.min_cn_ratio:
                        n_drop += 1
                        continue    # 脏碎片（数字/价格/URL等），不进模型
            elif not PP.keep_fragment(c, mode=args.filter_mode,
                                      min_ratio=args.min_cn_ratio,
                                      brand=brand, min_len=args.min_chunk):
                n_drop += 1
                continue
            if c not in uniq_idx:
                uniq_idx[c] = len(uniq)
                uniq.append(c)
            ids.append(uniq_idx[c])
        row_chunk_ids.append(ids)
        row_dropped.append(n_drop)

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

    # ---- 5) 聚合 + 判定
    n_labels = probs.shape[1] if probs is not None else 0
    # Opt-9：先固定 dev/test 划分。后续所有"从数据里学出来"的东西
    #（逐标签阈值 Opt-4、输出白名单 Opt-7）只准用 dev，test 只在报分时看一次。
    dev_idx, test_idx = PP.split_dev_test(len(data), args.n_dev, args.split_seed)
    gold_mat = np.zeros((len(data), n_labels), dtype=bool)
    for k, g in enumerate(row_golds):
        for lab in g:
            gold_mat[k, predictor.dataset.label_map[lab]] = True

    # 5a) 聚合分数（Opt-4：可换 noisy-OR；可选长帖归一化）
    agg_all = np.zeros((len(data), n_labels), dtype=np.float32)
    sup_all = np.zeros((len(data), n_labels), dtype=np.int32)
    row_ids_used = []
    for k, ids in enumerate(row_chunk_ids):
        if not ids:
            row_ids_used.append([])
            continue
        # Opt-5：同一句 OCR 在一帖内重复几十次，去重后再聚合/计票；
        # 否则重复片段既抬高 noisy-OR 分数、也刷高支持率票数
        use = sorted(set(ids)) if args.support_dedup else ids
        row_ids_used.append(use)
        sub = probs[use]
        agg_all[k] = PP.norm_long_doc(
            PP.agg_scores(sub, args.agg, args.agg_topk), len(use),
            args.norm_long_doc)
        sup_all[k] = (sub > threshold).sum(axis=0)

    # 5b) 阈值向量（Opt-4：逐标签阈值在 dev 上校准；否则全局单一阈值）
    if args.per_label_threshold:
        thr_map = PP.calibrate_per_label_threshold(
            agg_all[dev_idx], gold_mat[dev_idx], list(range(n_labels)),
            threshold, min_support=args.calib_min_support)
        thr_vec = np.array([thr_map[i] for i in range(n_labels)],
                           dtype=np.float32)
        n_cal = sum(1 for v in thr_map.values() if v != threshold)
        print(u"[Opt-4] 逐标签阈值：%d/%d 个标签在 dev(%d帖) 上完成校准，"
              u"其余回退全局阈值 %.2f；实际阈值范围 %.2f~%.2f"
              % (n_cal, n_labels, len(dev_idx), threshold,
                 thr_vec.min(), thr_vec.max()))
    else:
        thr_vec = np.full(n_labels, threshold, dtype=np.float32)

    # 5c) 票数向量（Opt-5：绝对票数按标签训练频率分档；否则沿用百分比支持率）
    votes_vec = None
    if args.support_votes > 0:
        dict_dir = getattr(conf.data, "dict_dir", "dict_yft")
        if not os.path.isabs(dict_dir):
            dict_dir = os.path.join(HERE, dict_dir)
        lab_freq = PP.load_label_freq(os.path.join(dict_dir, "doc_label.dict"))
        votes_vec = np.array(
            [PP.votes_needed(id_to_label[i], lab_freq, args.support_freq_split,
                             args.support_k_high, args.support_votes)
             for i in range(n_labels)], dtype=np.int32)
        print(u"[Opt-5] 绝对票数制：高频标签(训练频次>=%d)需 %d 票，长尾需 %d 票；"
              u"高频 %d 个 / 长尾 %d 个"
              % (args.support_freq_split, args.support_k_high,
                 args.support_votes,
                 int((votes_vec == args.support_k_high).sum()),
                 int((votes_vec != args.support_k_high).sum())))

    # 5d) Opt-7：输出标签白名单。从 dev 金标签统计，避免用 test 的标签分布
    # 反过来过滤 test 的预测（那是信息泄漏）；生产落地应改用业务确认的在用标签表。
    keep_label = np.ones(n_labels, dtype=bool)
    if args.label_whitelist:
        src_idx = dev_idx if args.label_whitelist_source == "dev" \
            else np.arange(len(data))
        wl = PP.build_output_whitelist([row_golds[i] for i in src_idx])
        wl_ids = [predictor.dataset.label_map[l] for l in wl
                  if l in predictor.dataset.label_map]
        keep_label = np.zeros(n_labels, dtype=bool)
        keep_label[wl_ids] = True
        if args.label_whitelist_source == "all":
            print(u"[Opt-7][警告] 白名单取自全量 %d 帖金标签 → 用 test 的标签分布过滤 "
                  u"test 的预测，属信息泄漏，结果只能当方案上限看，不可作为落地成绩"
                  % len(data))
        print(u"[Opt-7] 输出白名单（来源=%s，%d帖）：保留 %d 个标签，屏蔽其余 %d 个"
              u"（生产全量口径为 322 活跃 / 屏蔽 154 僵尸）"
              % (args.label_whitelist_source, len(src_idx),
                 int(keep_label.sum()), n_labels - int(keep_label.sum())))

    # 5e) Opt-3：检索兜底（片段精确命中训练文本 → 直接抄其标签，与模型取并集）
    retr_sets = None
    if args.retrieval_fallback:
        xlsx = args.train_xlsx if os.path.isabs(args.train_xlsx) \
            else os.path.join(HERE, args.train_xlsx)
        rmap = PP.build_retrieval_map(xlsx)
        retr_sets, n_frag_hit = [], 0
        for ids in row_ids_used:
            s = set()
            for i in set(ids):
                h = rmap.get(uniq[i])
                if h:
                    n_frag_hit += 1
                    s |= h
            retr_sets.append(set(l for l in s if l in vocab))
        print(u"[Opt-3] 检索兜底：训练映射 %d 条唯一片段；命中 %d 个片段，"
              u"%d 帖拿到兜底标签（均 %.1f 个/帖）"
              % (len(rmap), n_frag_hit, sum(1 for s in retr_sets if s),
                 sum(len(s) for s in retr_sets) / float(max(len(retr_sets), 1))))

    # 5f) 逐帖判定
    preds, pred_probs = [], []
    for k in range(len(data)):
        ids = row_ids_used[k]
        hit = {}
        if ids:
            agg, support = agg_all[k], sup_all[k]
            if votes_vec is not None:
                need_v = votes_vec                    # Opt-5 逐标签绝对票数
            elif args.min_support >= 1:
                need_v = np.full(n_labels, int(args.min_support), dtype=np.int32)
            else:
                nv = max(1, int(np.ceil(-args.min_support * len(ids))))
                if args.support_cap > 0:
                    # 长帖封顶：片段数越多 need 越大，会把只在少数片段出现的话题
                    # 整体压掉。实测长帖(>200片段)召回仅0.35、短帖(<=25)达0.80。
                    nv = min(nv, args.support_cap)
                need_v = np.full(n_labels, nv, dtype=np.int32)
            ok = np.where((agg > thr_vec) & (support >= need_v)
                          & keep_label)[0]
            if args.max_labels > 0 and len(ok) > args.max_labels:
                ok = ok[np.argsort(-agg[ok])[:args.max_labels]]
            hit = {id_to_label[int(i)]: float(agg[i]) for i in ok}
        if retr_sets:                                 # Opt-3 并集，作为下限保证
            for l in retr_sets[k]:
                hit.setdefault(l, 1.0)
        preds.append(set(hit.keys()))
        pred_probs.append(hit)

    probs_path = None
    if args.save_probs is not None:
        sp = resolve_out(args.save_probs, run_dir, "probs_agg_gold.npz")
        np.savez_compressed(sp, agg=agg_all, gold=gold_mat)
        probs_path = sp
        print(u"概率矩阵已保存: %s（可离线扫阈值）" % sp)
    chunk_probs_path = None
    if args.save_chunk_probs is not None and probs is not None:
        cp = resolve_out(args.save_chunk_probs, run_dir, "probs_chunk.npz")
        chunk_probs_path = cp
        counts = np.array([len(x) for x in row_chunk_ids], dtype=np.int32)
        flat = (np.concatenate([np.asarray(x, dtype=np.int32)
                                for x in row_chunk_ids])
                if counts.sum() else np.zeros(0, dtype=np.int32))
        np.savez_compressed(cp, chunk_probs=probs, row_counts=counts,
                            row_chunk_ids=flat, gold=gold_mat,
                            uniq_texts=np.array(uniq))
        print(u"片段级概率已保存: %s（%d 片段 × %d 标签，"
              u"调后处理规则可离线复算无需重跑推理）"
              % (cp, probs.shape[0], n_labels))

    # ---- 6) 指标
    # Opt-9：dev / test / all 三份都算出来，headline 只取 --split 指定的那份。
    # 在 all 上调参又在 all 上报分 = 用测试集调参，虚高不可信；这里强制分开。
    splits = [("dev", dev_idx), ("test", test_idx),
              ("all", np.arange(len(data)))]
    res = {}
    for nm, idx in splits:
        res[nm] = compute_metrics([row_golds[i] for i in idx],
                                  [preds[i] for i in idx], vocab_size)
    overall, label_stats = res[args.split]
    topk_desc = (u"top%d" % args.max_labels) if args.max_labels > 0 else u"不限"

    print(u"\n" + u"=" * 74)
    print(u"[Opt-9] dev/test 划分（seed=%d，dev=%d帖 只用于调参，test=%d帖 只用于报分）"
          % (args.split_seed, len(dev_idx), len(test_idx)))
    print(u"  %-6s %5s %8s %8s %8s %8s %8s"
          % (u"划分", u"帖数", u"microP", u"microR", u"microF1", u"macroF1",
             u"exactM"))
    for nm, _ in splits:
        o = res[nm][0]
        mark = u"  ← 本次 headline" if nm == args.split else u""
        print(u"  %-6s %5d %8.4f %8.4f %8.4f %8.4f %8.4f%s"
              % (nm, o["rows"], o["micro_P"], o["micro_R"], o["micro_F1"],
                 o["macro_F1"], o["exact_match"], mark))
    print(u"=" * 74)
    print(u"%s 集 %d 条的总指标（%s，阈值=%.2f，每行标签上限=%s）"
          % (args.split, overall["rows"], conf.model_name, threshold,
             topk_desc))
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
        u"切分诊断": {
            u"切分出片段数": int(sum(row_raw_n)),
            u"被过滤丢弃数": int(sum(row_dropped)),
            u"过滤丢弃率": round(sum(row_dropped) /
                              float(max(sum(row_raw_n), 1)), 4),
            u"保留片段数": n_chunks,
            u"保留片段去重率": round(1 - len(uniq) / float(max(n_chunks, 1)), 4),
            u"每帖保留片段_中位": int(np.median([len(x) for x in row_chunk_ids])),
            u"无有效片段的帖数": int(sum(1 for x in row_chunk_ids if not x)),
        },
        u"总指标": overall,
        u"分划分指标": {nm: res[nm][0] for nm in ("dev", "test", "all")},
        u"产物文件": {
            u"逐行结果": out_path,
            u"逐标签指标": pl_path,
            u"运行日志": os.path.join(run_dir, "run.log"),
            u"概率矩阵": probs_path,
            u"片段级概率矩阵": chunk_probs_path,
        },
        u"后处理描述": {
            u"聚合方式": args.agg + (u"(k=%d)" % args.agg_topk
                                     if args.agg == u"topk" else u""),
            u"支持率need": (u"固定%d" % int(args.min_support)
                            if args.min_support >= 1 else
                            u"ceil(%.3f×片段数)%s" % (
                                -args.min_support,
                                u"，上限%d" % args.support_cap
                                if args.support_cap > 0 else u"，不封顶")),
            u"每帖标签上限": topk_desc,
            u"已开启的文档Opt": (lambda: [
                x for x, on in [
                    (u"Opt-1白名单脏过滤", args.filter_mode == u"whitelist"),
                    (u"Opt-3检索兜底", args.retrieval_fallback),
                    (u"Opt-4 noisy-OR聚合", args.agg == u"noisyor"),
                    (u"Opt-4逐标签阈值", args.per_label_threshold),
                    (u"Opt-4长帖归一化", args.norm_long_doc),
                    (u"Opt-5去重计票", args.support_dedup),
                    (u"Opt-5绝对票数", args.support_votes > 0),
                    (u"Opt-7输出标签白名单", args.label_whitelist),
                    (u"Opt-8小数点保护", args.protect_decimal),
                    (u"v2支持率封顶cap%d" % args.support_cap,
                     args.support_cap > 0),
                ] if on] or [u"无（全部默认=旧脚本行为）"])(),
            u"评测划分": args.split,
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
