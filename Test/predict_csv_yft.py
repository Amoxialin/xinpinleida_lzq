#!/usr/bin/env python
# coding: utf-8
"""Batch label prediction for a CSV export -> Excel report.

在 predict_one.py 的基础上做批量版：
  * 复用它的分段逻辑 split_chunks（换行 -> 标点 -> 滑动窗口，步长=chunk/2）
  * 复用它的聚合逻辑（每个标签取所有片段的最大概率）
  * 新增：CSV 读取（自动识别编码/分隔符/表头）、全局片段去重、分批 GPU 推理、Excel 输出

Examples:
    python3 predict_csv_yft.py --csv data_export_sample_1100.csv --preview
    python3 predict_csv_yft.py --csv data_export_sample_1100.csv --out result.xlsx
    python3 predict_csv_yft.py --csv x.csv --text-cols F,I --id-col B --threshold 0.3
    python3 predict_csv_yft.py --csv x.csv --text-cols 标题,正文 --id-col 帖子ID
"""

import argparse
import codecs
import csv
import io
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from predict_one import build_line, split_chunks   # noqa: E402  复用分段/打包

# Excel 单元格不接受的控制字符 & 长度上限
ILLEGAL_XLSX_RE = re.compile(u"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]")
CELL_LIMIT = 32000
WS_RE = re.compile(u"\\s+")

ID_HINTS = ("id", "帖子id", "笔记id", "内容id", "postid", "noteid", "feedid",
            "docid", "mid", "编号")
# 含这些词的列即使带 id 也不是帖子ID（item_id / uid / head_url 之类）
ID_BLOCK = ("item", "product", "goods", "sku", "spu", "user", "uid", "author",
            "order", "head", "url", "robot", "type")


def col_letter(i):
    """0-based 下标 -> Excel 列字母（0->A, 26->AA）。"""
    s, n = "", i + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ---------------------------------------------------------------- CSV 读取

def decode_bytes(raw):
    """自动识别编码，返回 (文本, 编码名)。"""
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    for enc in ("utf-8", "gb18030", "big5"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def sniff_delimiter(text):
    sample = text[:16384]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    lines = sample.splitlines()
    if not lines:
        return ","
    first = lines[0]
    best = max(",;\t|", key=lambda d: first.count(d))
    return best if first.count(best) > 0 else ","


def raise_field_limit():
    """导出文件里单条 content/OCR 可能超过 csv 默认 128KB 上限。"""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit = int(limit // 10)


def read_csv(path, encoding=None, delimiter=None):
    raise_field_limit()
    with open(path, "rb") as fin:
        raw = fin.read()
    text, used_enc = decode_bytes(raw)
    if encoding:
        text, used_enc = raw.decode(encoding, errors="replace"), encoding
    used_delim = delimiter or sniff_delimiter(text)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=used_delim)]
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    return rows, used_enc, used_delim


def guess_has_header(rows):
    """启发式判断首行是否为表头。"""
    if len(rows) < 2:
        return True
    r0, r1 = rows[0], rows[1]

    def profile(r):
        long_cells = sum(1 for c in r if len(str(c)) > 40)
        digit_cells = sum(1 for c in r if str(c).strip().isdigit())
        return long_cells, digit_cells

    l0, d0 = profile(r0)
    l1, _ = profile(r1)
    if l0 == 0 and l1 > 0:          # 首行都很短、次行有长文本 -> 表头
        return True
    if d0 >= 1 and l0 >= 1:         # 首行就有长数字ID + 长文本 -> 数据行
        return False
    return True


def col_to_index(spec, header):
    """列名 / Excel列字母(F) / 1-based序号(6) -> 0-based下标。"""
    s = str(spec).strip()
    if header:
        for i, h in enumerate(header):
            if h and str(h).strip() == s:
                return i
    if re.fullmatch(r"[A-Za-z]{1,3}", s):
        idx = 0
        for ch in s.upper():
            idx = idx * 26 + (ord(ch) - 64)
        return idx - 1
    if re.fullmatch(r"\d+", s):
        return int(s) - 1
    raise ValueError(u"无法解析列: %s（可用列名/字母/序号）" % spec)


def detect_id_col(header):
    if not header:
        return 0
    norm = [re.sub(r"[\s_\-]", "", str(h or "")).lower() for h in header]
    ok = [i for i, n in enumerate(norm)
          if n and not any(b in n for b in ID_BLOCK)]
    for hint in ID_HINTS:                     # 先精确匹配（"id" 优先）
        for i in ok:
            if norm[i] == hint:
                return i
    for i in ok:                              # 再退化成包含匹配
        if "id" in norm[i] or u"编号" in norm[i]:
            return i
    return 0


# ---------------------------------------------------------------- 推理

def chunkify(text, args, tokenize, max_token_len):
    """与 predict_one.run 完全一致的分段规则。"""
    toks = tokenize(text)
    if not toks:
        return []
    if args.no_split or len(toks) <= max(16, args.chunk):
        return [text]
    return split_chunks(text, args.chunk, min_len=args.min_chunk)


def batch_predict(predictor, uniq_list, tokenize, batch_size):
    """分批前向，返回 (N, num_labels) float32 概率矩阵。"""
    if not uniq_list:
        return np.zeros((0, 0), dtype=np.float32)
    out = None
    t0 = time.time()
    total = len(uniq_list)
    for start in range(0, total, batch_size):
        part = uniq_list[start:start + batch_size]
        probs = predictor.predict([build_line(t, tokenize) for t in part])
        if out is None:
            out = np.zeros((total, probs.shape[1]), dtype=np.float32)
        out[start:start + len(part)] = probs
        done = start + len(part)
        if done % (batch_size * 10) < batch_size or done == total:
            print(u"  推理 %d/%d 片段  %.1fs" % (done, total, time.time() - t0))
    return out


# ---------------------------------------------------------------- 输出

def clean_cell(text, limit=CELL_LIMIT):
    if text is None:
        return ""
    s = ILLEGAL_XLSX_RE.sub("", str(text))
    return s if len(s) <= limit else s[:limit] + u"…(截断)"


def cell_value(v, limit=CELL_LIMIT):
    """数字按数值写入（Excel 里可排序/求和），字符串才做清洗与截断。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return clean_cell(v, limit)
    return v


def build_row_result(ids, probs, chunks_of_ids, id_to_label, threshold, top_n,
                     strong=0.6, max_labels=0, min_support=1):
    """把某行的若干片段概率聚合成一行结果。

    agg      : 每个标签在所有片段上的最大概率（与 predict_one.py 一致）
    support  : 有多少个片段的该标签概率过了阈值 —— 用来压单窗口噪声
               （长文里 "满20可用" 这种 OCR 碎片偶尔命中一次，support=1）
    min_support: >=1 视为片段个数；<1 视为占总片段的比例
    """
    sub = probs[ids]
    k = sub.shape[0]
    agg = sub.max(axis=0)
    best = sub.argmax(axis=0)
    support = (sub > threshold).sum(axis=0)
    need = min_support if min_support >= 1 else max(1, int(-(-min_support * k // 1)))
    order = np.argsort(-agg)

    all_hits = [(id_to_label[int(i)], float(agg[i]), int(best[i]),
                 int(support[i])) for i in order if agg[i] > threshold]
    hits = [h for h in all_hits if h[3] >= need]
    if max_labels > 0:
        hits = hits[:max_labels]

    top = [(id_to_label[int(i)], float(agg[i])) for i in order[:top_n]]
    evidence = u"; ".join(
        u"%s←\"%s\"" % (name, chunks_of_ids[ci][:40])
        for name, _, ci, _s in hits[:8])
    detail = u"; ".join(u"%s %.4f(%d/%d)" % (n, p, s, k)
                        for n, p, _, s in hits)
    top_str = u"; ".join(u"%s%s %.3f" % (u"*" if p > threshold else u"", n, p)
                         for n, p in top)
    return {
        "labels": [n for n, _, _, _ in hits],
        "strong": [n for n, p, _, s in all_hits if p >= strong and s >= need],
        "n_all_hits": len(all_hits),
        "detail": detail,
        "evidence": evidence,
        "top": top_str,
        "num_chunks": k,
    }


def write_xlsx(path, header, rows, widths, summary):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = u"预测结果"
    ws.append(header)
    fill = PatternFill("solid", fgColor="DDEBF7")
    for c, _ in enumerate(header, 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for r in rows:
        ws.append([cell_value(v) for v in r])
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    ws2 = wb.create_sheet(u"运行信息")
    ws2.append([u"项", u"值"])
    for c in (1, 2):
        ws2.cell(row=1, column=c).font = Font(bold=True)
    for k, v in summary:
        ws2.append([cell_value(k), cell_value(v)])
    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 80
    wb.save(path)


def write_csv_out(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header)
        for r in rows:
            w.writerow([cell_value(v, 20000) for v in r])


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description=u"CSV 批量标签预测 -> Excel（基于 predict_one.py）")
    parser.add_argument("--csv", required=True, help=u"输入 CSV 路径")
    parser.add_argument("--out", default=None,
                        help=u"输出文件，.xlsx（默认）或 .csv")
    parser.add_argument("--conf", default="conf/train.yft.json")
    parser.add_argument("--model", default=None, help=u"覆盖 conf.eval.model_dir")
    parser.add_argument("--text-cols", default="F,I",
                        help=u"文本所在列，逗号分隔；支持 Excel字母/序号/列名（默认 F,I）")
    parser.add_argument("--id-col", default=None,
                        help=u"帖子ID列；默认自动识别，识别不到用第一列")
    parser.add_argument("--encoding", default=None, help=u"强制指定编码，如 gbk")
    parser.add_argument("--delimiter", default=None, help=u"强制指定分隔符")
    parser.add_argument("--header", default="auto", choices=["auto", "yes", "no"])
    parser.add_argument("--threshold", type=float, default=None,
                        help=u"默认使用 conf.eval.threshold")
    parser.add_argument("--chunk", type=int, default=32,
                        help=u"长文本滑窗长度（默认32）")
    parser.add_argument("--min-chunk", type=int, default=2,
                        help=u"丢弃短于该长度的碎片")
    parser.add_argument("--top", type=int, default=10,
                        help=u"结果里附带 Top-N 候选（含未过阈值）")
    parser.add_argument("--strong", type=float, default=0.6,
                        help=u"「强信号标签」的概率门槛（默认0.6）")
    parser.add_argument("--min-support", type=float, default=1,
                        help=u"标签至少要在几个片段上过阈值才算命中；"
                             u">=1 是片段个数，<1 是占片段总数的比例（默认1=不过滤）")
    parser.add_argument("--max-labels", type=int, default=0,
                        help=u"每行最多输出几个标签，按概率取前 N（0=不限）")
    parser.add_argument("--no-split", action="store_true",
                        help=u"关闭长文本分段（只按 max_token_len 截断）")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help=u"只跑前 N 行，0=全部")
    parser.add_argument("--preview", action="store_true",
                        help=u"只打印列结构和样例，不加载模型")
    args = parser.parse_args()

    conf_path = args.conf if os.path.isabs(args.conf) else os.path.abspath(args.conf)
    os.chdir(HERE)

    if not os.path.exists(args.csv):
        sys.exit(u"[错误] 找不到文件: %s" % args.csv)

    rows, enc, delim = read_csv(args.csv, args.encoding, args.delimiter)
    if not rows:
        sys.exit(u"[错误] CSV 是空的")

    has_header = (args.header == "yes") or \
                 (args.header == "auto" and guess_has_header(rows))
    header = rows[0] if has_header else None
    data = rows[1:] if has_header else rows
    if args.limit > 0:
        data = data[:args.limit]

    ncols = max(len(r) for r in rows)
    print(u"CSV: %s" % args.csv)
    print(u"  编码=%s 分隔符=%r 总行数=%d 列数=%d 表头=%s 数据行=%d"
          % (enc, delim, len(rows), ncols, has_header, len(data)))
    if header:
        print(u"  表头: %s" % u" | ".join(
            u"%s:%s" % (col_letter(i), h) for i, h in enumerate(header)))

    text_specs = [s for s in args.text_cols.split(",") if s.strip()]
    text_idx = [col_to_index(s, header) for s in text_specs]
    id_idx = col_to_index(args.id_col, header) if args.id_col \
        else detect_id_col(header)
    for i in text_idx + [id_idx]:
        if i < 0 or (header and i >= len(header)) or i >= ncols:
            sys.exit(u"[错误] 列下标越界: %d（该文件共 %d 列）" % (i, ncols))

    def col_desc(i):
        name = str(header[i]).strip() if header and i < len(header) else u""
        return u"第%d列(%s%s)" % (i + 1, col_letter(i),
                                 (u": " + name) if name else u"")

    print(u"  ID列 = %s" % col_desc(id_idx))
    print(u"  文本列 = %s" % u", ".join(col_desc(i) for i in text_idx))

    def cell(row, i):
        return str(row[i]).strip() if i < len(row) and row[i] is not None else ""

    print(u"\n前3行预览:")
    for r in data[:3]:
        print(u"  ID=%s" % cell(r, id_idx))
        for spec, i in zip(text_specs, text_idx):
            t = cell(r, i)
            print(u"    [%s] (%d字) %s" % (spec, len(t),
                                           t[:60].replace("\n", " ")))

    if args.preview:
        return

    from config import Config
    from predict import Predictor
    from prepare_yft_data import tokenize

    conf = Config(config_file=conf_path)
    if args.model:
        conf.eval.model_dir = args.model
    threshold = conf.eval.threshold if args.threshold is None else args.threshold
    max_token_len = conf.feature.max_token_len

    print(u"\nloading model: %s ..." % conf.eval.model_dir)
    predictor = Predictor(conf)
    id_to_label = {v: k for k, v in predictor.dataset.label_map.items()}
    print(u"ready. labels=%d threshold=%.2f chunk=%d min_chunk=%d split=%s"
          % (len(id_to_label), threshold, args.chunk, args.min_chunk,
             not args.no_split))

    # ---- 1) 逐行切片段 + 全局去重
    uniq, uniq_idx = [], {}
    row_ids, row_chunks = [], []
    row_texts = []
    for r in data:
        parts = [cell(r, i) for i in text_idx]
        text = u"\n".join(p for p in parts if p)
        row_texts.append((parts, text))
        ids = []
        for c in chunkify(text, args, tokenize, max_token_len):
            if not tokenize(c):
                continue
            if c not in uniq_idx:
                uniq_idx[c] = len(uniq)
                uniq.append(c)
            ids.append(uniq_idx[c])
        row_ids.append(ids)
        row_chunks.append([uniq[i] for i in ids])

    n_chunks = sum(len(x) for x in row_ids)
    print(u"\n%d 行 -> %d 个片段，去重后 %d 个待预测"
          % (len(data), n_chunks, len(uniq)))
    empty_rows = sum(1 for x in row_ids if not x)
    if empty_rows:
        print(u"  其中 %d 行没有可识别的中文/字母数字内容，将标记为「无有效文本」"
              % empty_rows)

    # ---- 2) 分批推理
    t0 = time.time()
    probs = batch_predict(predictor, uniq, tokenize, args.batch_size)
    infer_sec = time.time() - t0
    print(u"推理完成: %.1fs（%.0f 片段/秒）"
          % (infer_sec, len(uniq) / max(infer_sec, 1e-6)))

    # ---- 3) 聚合 + 组装 Excel 行
    n_text = len(text_idx)
    text_col_titles = []
    for i in text_idx:
        name = str(header[i]).strip() if header and i < len(header) else u""
        text_col_titles.append(u"%s(%s列)" % (name or u"文本", col_letter(i)))
    out_header = [u"帖子ID"] + text_col_titles + [
        u"合并文本", u"字数", u"片段数", u"标签数", u"预测标签",
        u"强信号标签(≥%.2f)" % args.strong,
        u"标签-概率(命中片段数)", u"证据片段", u"Top%d候选" % args.top]
    label_col = n_text + 5
    out_rows = []
    label_counter = {}
    strong_counter = {}
    n_label_list = []
    n_hit = 0
    for r, (parts, text), ids, chunks in zip(data, row_texts, row_ids, row_chunks):
        pid = cell(r, id_idx)
        n_chars = len(WS_RE.sub("", text))
        if not ids:
            res = {"labels": [], "strong": [], "n_all_hits": 0, "detail": u"",
                   "evidence": u"", "top": u"", "num_chunks": 0}
            note = u"无有效文本"
        else:
            res = build_row_result(ids, probs, chunks, id_to_label, threshold,
                                   args.top, args.strong, args.max_labels,
                                   args.min_support)
            note = u""
        n_label_list.append(len(res["labels"]))
        if res["labels"]:
            n_hit += 1
            for lab in res["labels"]:
                label_counter[lab] = label_counter.get(lab, 0) + 1
            for lab in res["strong"]:
                strong_counter[lab] = strong_counter.get(lab, 0) + 1
        vals = (list(parts) + [u""] * n_text)[:n_text]
        out_rows.append(
            [pid] + vals + [
                text, n_chars, res["num_chunks"], len(res["labels"]),
                u",".join(res["labels"]) or note,
                u",".join(res["strong"]), res["detail"],
                res["evidence"], res["top"],
            ])

    widths = [24] + [36] * n_text + [50, 8, 8, 8, 34, 30, 52, 46, 50]
    out_path = args.out or os.path.splitext(os.path.basename(args.csv))[0] + \
        u"_labels.xlsx"
    if not os.path.isabs(out_path):
        out_path = os.path.join(HERE, out_path)

    summary = [
        (u"输入CSV", args.csv),
        (u"编码/分隔符", u"%s / %r" % (enc, delim)),
        (u"数据行数", len(data)),
        (u"ID列", col_desc(id_idx)),
        (u"文本列", u", ".join(col_desc(i) for i in text_idx)),
        (u"模型", conf.eval.model_dir),
        (u"模型结构", conf.model_name),
        (u"标签总数", len(id_to_label)),
        (u"判定阈值", threshold),
        (u"强信号阈值", args.strong),
        (u"最小命中片段数", args.min_support),
        (u"每行标签上限", args.max_labels or u"不限"),
        (u"滑窗长度/步长", u"%d / %d" % (args.chunk, max(1, args.chunk // 2))),
        (u"最小片段长度", args.min_chunk),
        (u"长文本分段", u"关闭" if args.no_split else u"开启"),
        (u"片段总数/去重后", u"%d / %d" % (n_chunks, len(uniq))),
        (u"无有效文本行数", empty_rows),
        (u"至少命中1个标签的行数", u"%d (%.1f%%)"
         % (n_hit, 100.0 * n_hit / max(len(data), 1))),
        (u"平均标签数/行", u"%.2f" % (sum(label_counter.values())
                                     / max(len(data), 1))),
        (u"标签数分布 中位/P90/最大", u"%d / %d / %d" % (
            int(np.median(n_label_list)), int(np.percentile(n_label_list, 90)),
            int(max(n_label_list))) if n_label_list else u"-"),
        (u"平均强信号标签数/行", u"%.2f" % (sum(strong_counter.values())
                                           / max(len(data), 1))),
        (u"推理耗时", u"%.1f 秒" % infer_sec),
        (u"生成时间", time.strftime("%Y-%m-%d %H:%M:%S")),
        (u"Top20高频预测标签", u"; ".join(
            u"%s:%d" % kv for kv in sorted(label_counter.items(),
                                           key=lambda x: -x[1])[:20])),
        (u"Top20高频强信号标签", u"; ".join(
            u"%s:%d" % kv for kv in sorted(strong_counter.items(),
                                           key=lambda x: -x[1])[:20])),
    ]

    if out_path.lower().endswith(".csv"):
        write_csv_out(out_path, out_header, out_rows)
    else:
        write_xlsx(out_path, out_header, out_rows, widths, summary)

    print(u"\n已写出: %s" % out_path)
    print(u"  %d 行，其中 %d 行至少命中1个标签（阈值 %.2f，强信号 %.2f，"
          u"min-support %s）"
          % (len(out_rows), n_hit, threshold, args.strong, args.min_support))
    if n_label_list:
        print(u"  每行标签数 中位/P90/最大 = %d / %d / %d，平均 %.1f；"
              u"强信号标签平均 %.1f 个/行"
              % (int(np.median(n_label_list)),
                 int(np.percentile(n_label_list, 90)), max(n_label_list),
                 sum(n_label_list) / float(len(n_label_list)),
                 sum(strong_counter.values()) / float(max(len(data), 1))))
    print(u"  Top10 高频预测标签: %s" % u", ".join(
        u"%s(%d)" % kv for kv in sorted(label_counter.items(),
                                        key=lambda x: -x[1])[:10]))
    print(u"  Top10 高频强信号标签: %s" % u", ".join(
        u"%s(%d)" % kv for kv in sorted(strong_counter.items(),
                                        key=lambda x: -x[1])[:10]))
    print(u"\n前5行结果预览:")
    for row in out_rows[:5]:
        print(u"  %-20s -> %s" % (row[0], row[label_col] or u"(无)"))


if __name__ == "__main__":
    main()
