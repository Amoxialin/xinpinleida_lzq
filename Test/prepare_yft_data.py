#!/usr/bin/env python
# coding: utf-8
"""Convert data/yft.jsonl + data/yft.txt into NeuralClassifier json format.

Input line format of yft.jsonl:
    {"content": "珀莱雅", "label": ["品牌提及", "国货品牌"]}

Output line format (required by dataset/classification_dataset.py):
    {"doc_label": [...], "doc_token": [...], "doc_keyword": [], "doc_topic": []}

Tokenization is char-level: every CJK char becomes one token, every
consecutive ascii letter/digit run becomes one token, punctuation and
whitespace are dropped. This avoids needing a Chinese word segmenter and
fits the very short texts of this dataset.

Splitting is done on unique content (all rows sharing the same content go to
the same split) so that identical texts never leak across train/dev/test.
Every label is guaranteed to appear at least once in the train split.
"""

import argparse
import collections
import json
import os
import random
import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def normalize(text):
    return unicodedata.normalize("NFKC", text).strip()


def tokenize(text):
    return TOKEN_RE.findall(normalize(text).lower())


def load_labels(path):
    labels = []
    seen = set()
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            label = normalize(line).strip('"').strip("'").strip()
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def load_records(path, valid_labels):
    """Group rows by content, keeping each row's label list."""
    groups = collections.OrderedDict()
    stats = collections.Counter()
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                stats["bad_json"] += 1
                continue
            content = normalize(obj.get("content", ""))
            labels = obj.get("label", [])
            if isinstance(labels, str):
                labels = [labels]
            labels = [normalize(l).strip('"') for l in labels]
            labels = [l for l in labels if l]
            if not content:
                stats["empty_content"] += 1
                continue
            if not labels:
                stats["empty_label"] += 1
                continue
            unknown = [l for l in labels if l not in valid_labels]
            if unknown:
                stats["unknown_label_rows"] += 1
                stats["unknown_label_hits"] += len(unknown)
            tokens = tokenize(content)
            if not tokens:
                stats["empty_after_tokenize"] += 1
                continue
            groups.setdefault(content, []).append(labels)
            stats["rows"] += 1
    return groups, stats


def dedup_groups(groups, mode, vote_ratio=0.5):
    """Collapse repeated rows of the same content.

    mode:
      none  - keep every row as-is (original behaviour)
      exact - drop rows whose label set is identical to an earlier row
      vote  - one row per content; keep labels seen in >= vote_ratio of rows
      union - one row per content; keep every label seen in any row
    """
    if mode == "none":
        return groups, {}
    out = collections.OrderedDict()
    info = collections.Counter()
    for content, rows in groups.items():
        if mode == "exact":
            seen, kept = set(), []
            for labels in rows:
                key = tuple(sorted(labels))
                if key in seen:
                    info["dropped_duplicate_rows"] += 1
                    continue
                seen.add(key)
                kept.append(labels)
            out[content] = kept
            continue

        n = len(rows)
        freq = collections.Counter()
        for labels in rows:
            freq.update(set(labels))
        if mode == "union":
            merged = sorted(freq)
        else:
            need = max(1, int(-(-(n * vote_ratio) // 1)))   # ceil
            merged = sorted(l for l, c in freq.items() if c >= need)
        if not merged:
            merged = sorted(collections.Counter(
                tuple(sorted(l)) for l in rows).most_common(1)[0][0])
            info["vote_fallback_to_most_common"] += 1
        if len({tuple(sorted(l)) for l in rows}) > 1:
            info["conflict_contents"] += 1
            info["rows_merged"] += n - 1
        out[content] = [merged]
    return out, dict(info)


def split_groups(groups, dev_ratio, test_ratio, seed):
    """Assign each unique content to train/dev/test."""
    contents = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(contents)
    num_test = max(1, int(round(len(contents) * test_ratio)))
    num_dev = max(1, int(round(len(contents) * dev_ratio)))
    assignment = {}
    for content in contents[:num_test]:
        assignment[content] = "test"
    for content in contents[num_test:num_test + num_dev]:
        assignment[content] = "dev"
    for content in contents[num_test + num_dev:]:
        assignment[content] = "train"
    return assignment


def ensure_labels_in_train(groups, assignment, rng):
    """Move one group per train-missing label from dev/test into train."""
    train_labels = collections.Counter()
    for content, split in assignment.items():
        if split == "train":
            for labels in groups[content]:
                train_labels.update(labels)
    missing = []
    all_labels = set()
    for content in groups:
        for labels in groups[content]:
            all_labels.update(labels)
    moved = 0
    for label in sorted(all_labels - set(train_labels)):
        if train_labels[label] > 0:
            continue
        donor = None
        for content in groups:
            if assignment[content] == "train":
                continue
            if any(label in labels for labels in groups[content]):
                donor = content
                break
        if donor is None:
            missing.append(label)
            continue
        assignment[donor] = "train"
        for labels in groups[donor]:
            train_labels.update(labels)
        moved += 1
    return moved, missing


def write_split(groups, assignment, split, out_path):
    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for content, rows in groups.items():
            if assignment[content] != split:
                continue
            tokens = tokenize(content)
            for labels in rows:
                fout.write(json.dumps({
                    "doc_label": labels,
                    "doc_token": tokens,
                    "doc_keyword": [],
                    "doc_topic": []
                }, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/yft.jsonl")
    parser.add_argument("--labels", default="data/yft.txt")
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--prefix", default="yft")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedup", default="none",
                        choices=["none", "exact", "vote", "union"],
                        help="none=原样保留; exact=删除完全重复行; "
                             "vote=同文本按标签多数投票合并成一行; "
                             "union=同文本取所有标签的并集")
    parser.add_argument("--vote-ratio", type=float, default=0.5,
                        help="--dedup vote 时，标签需在该文本的行中出现比例>=此值")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    groups, stats = load_records(args.input, set(labels))
    rows_before = sum(len(r) for r in groups.values())
    groups, dedup_info = dedup_groups(groups, args.dedup, args.vote_ratio)
    rows_after = sum(len(r) for r in groups.values())
    rng = random.Random(args.seed)
    assignment = split_groups(groups, args.dev_ratio, args.test_ratio, args.seed)
    moved, missing = ensure_labels_in_train(groups, assignment, rng)

    paths = {s: os.path.join(args.outdir, "%s_%s.json" % (args.prefix, s))
             for s in ("train", "dev", "test")}
    counts = {s: write_split(groups, assignment, s, paths[s]) for s in paths}

    token_counts = collections.Counter()
    lengths = []
    label_counts = collections.Counter()
    for content, rows in groups.items():
        tokens = tokenize(content)
        lengths.append(len(tokens))
        if assignment[content] == "train":
            token_counts.update(set(tokens))
            for l in rows:
                label_counts.update(l)
    lengths.sort()

    def pct(p):
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))]

    print("label vocab (yft.txt): %d" % len(labels))
    print("rows kept: %d, unique contents: %d" % (stats["rows"], len(groups)))
    print("dedup mode: %s (vote_ratio=%s)  rows %d -> %d (removed %d) %s"
          % (args.dedup, args.vote_ratio, rows_before, rows_after,
             rows_before - rows_after,
             json.dumps(dedup_info, ensure_ascii=False)))
    print("skipped: %s" % json.dumps(
        {k: v for k, v in stats.items() if k != "rows"}, ensure_ascii=False))
    print("groups moved to train for label coverage: %d, still missing: %d %s"
          % (moved, len(missing), missing[:10]))
    print("token len (chars): p50=%d p90=%d p99=%d max=%d"
          % (pct(0.50), pct(0.90), pct(0.99), lengths[-1]))
    print("distinct tokens in train: %d" % len(token_counts))
    print("distinct labels in train: %d" % len(label_counts))
    for s in ("train", "dev", "test"):
        print("%-5s -> %s (%d lines)" % (s, paths[s], counts[s]))


if __name__ == "__main__":
    main()
