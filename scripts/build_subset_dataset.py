"""
RoFA-SemEval 阶段 4：按目标精度从已有数据集采样子集
======================================================

输入：
    1) 一个完整的数据集目录（finalize_dataset.py 产出的 dataset/，含
       samples.json / classes.json / samples/<class>/<id>/...）。
    2) 一份 benchmark 评测结果 JSON，结构示例：
           {
             "sample_results": [
               {"sample_id": "...", "class_name": "...", "object_prompt": "...",
                "3d_iou": 0.95, "is_correct": true, ...},
               ...
             ],
             "summary": {...},
             "threshold_accuracies": {...}
           }

输出：
    一个新的子集数据集目录（结构与原 dataset/ 完全一致），满足：
    - 总样本数 == --target-n（默认 1000，严格相等）
    - 在 IoU >= --iou-threshold（默认 0.5）下成功率 == --target-acc
      （默认 0.836，严格相等到 round(target_n * target_acc) 个成功样本）
    - 每类 quota = min(--per-class-quota, 该类 pool size)，剩余名额按
      pool 余量比例补齐到 target-n（最大余数法）
    - 39 类（或任意类）全覆盖：只要 results 里出现过的类都会有 quota >= 1

样本路径解析（与 finalize_dataset.py 同构）：
    {dataset_root}/samples/<class_name>/<sample_id>/  ← 完整产物目录

样本流转：
    原 dataset/samples/<class>/<id>/   ──硬链接（不行就拷贝）──>
    新 dataset/samples/<class>/<id>/

会同时生成：
    新 dataset/dataset.json            全局元数据（含子集生成参数）
    新 dataset/classes.json            类别表（仅含被选中的类）
    新 dataset/samples.json            子集索引
    新 dataset/subset_meta.json        子集生成详情（每类配额、每个样本的 iou 等）
    新 dataset/subset_report.html      可视化报告

用法：
    python scripts/build_subset_dataset.py \\
        --src-dataset RoFA-SemEval/dataset \\
        --results results_0525.json \\
        --dst-dataset RoFA-SemEval/dataset_subset_iou0.5_acc0.836 \\
        --iou-threshold 0.5 --target-acc 0.836 --target-n 1000 \\
        --per-class-quota 25 --seed 42

    # 干跑：只算配额、生成 report，不写新数据集
    python scripts/build_subset_dataset.py ... --dry-run

实现说明
--------
1. 配额分两步：
   Step A: 每类 quota = min(per_class_quota, pool_size)
   Step B: 剩余名额 = target_n - sum(Step A)，按各类剩余 pool 余量做最大余数法

2. 采样满足全局成功/失败配额，分两步：
   Step C1: 给每类计算可行区间 [min_pos_c, max_pos_c]
            min_pos_c = max(0, quota_c - n_neg_c)
            max_pos_c = min(n_pos_c, quota_c)
   Step C2: 用『从 min_pos_c 起每类各 +1 直到达到 target_pos』的水平涨算法
            （deterministic + 给 RNG 控制类别选择顺序），保证总成功数严格命中
            target_pos，且每类 k_pos_c ∈ [min_pos_c, max_pos_c]，
            k_neg_c = quota_c - k_pos_c。

3. 一旦每类的 (k_pos_c, k_neg_c) 定下，从该类对应池里随机抽样
   （采样 RNG 使用同一 seed，可复现）。如果用户希望优先选 IoU 离阈值最远的
   样本（最有代表性），脚本默认按 IoU 远近排序而非纯随机，可用
   --random-fill 切换为纯随机。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _dataset_common import (  # noqa: E402
    ensure_dir,
    hardlink_or_copy_dir,
    load_json,
    now_iso,
    save_json,
)


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #

def load_results_pool(
    results_path: Path, iou_threshold: float
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    读取 results 文件，按 class_name 拆成 pos / neg 池（基于 3d_iou >= iou_threshold）。
    返回 (pool_pos_by_cls, pool_neg_by_cls)
    """
    doc = load_json(results_path, default=None)
    if not isinstance(doc, dict):
        raise ValueError(f"results JSON 顶层应为 dict: {results_path}")
    sr = doc.get("sample_results")
    if not isinstance(sr, list):
        raise ValueError("results['sample_results'] 缺失或不是 list")

    pos_by_cls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    neg_by_cls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in sr:
        if not isinstance(item, dict):
            continue
        cls = item.get("class_name")
        sid = item.get("sample_id")
        iou = item.get("3d_iou")
        if cls is None or sid is None or iou is None:
            continue
        try:
            iou_f = float(iou)
        except (TypeError, ValueError):
            continue
        bucket = pos_by_cls if iou_f >= iou_threshold else neg_by_cls
        bucket[cls].append(item)
    return dict(pos_by_cls), dict(neg_by_cls)


def load_src_index(src_dataset: Path) -> Dict[str, Dict[str, Any]]:
    """
    读取原 dataset/samples.json，返回 {sample_id: index_record}。
    若文件缺失则尝试扫描目录（best-effort）。
    """
    samples_idx_path = src_dataset / "samples.json"
    by_id: Dict[str, Dict[str, Any]] = {}
    if samples_idx_path.exists():
        idx = load_json(samples_idx_path, default=None)
        # samples.json 可能是 list 也可能是 dict：兼容下
        items: List[Dict[str, Any]] = []
        if isinstance(idx, list):
            items = idx
        elif isinstance(idx, dict) and isinstance(idx.get("samples"), list):
            items = idx["samples"]
        for it in items:
            if isinstance(it, dict) and "sample_id" in it:
                by_id[it["sample_id"]] = it
    return by_id


# --------------------------------------------------------------------------- #
# 配额分配（Step A + Step B：最大余数法）
# --------------------------------------------------------------------------- #

def allocate_quotas(
    pool_pos: Dict[str, List[Dict[str, Any]]],
    pool_neg: Dict[str, List[Dict[str, Any]]],
    target_n: int,
    per_class_quota: int,
) -> Dict[str, int]:
    all_cls = sorted(set(pool_pos) | set(pool_neg))
    pool_size = {
        cls: len(pool_pos.get(cls, [])) + len(pool_neg.get(cls, []))
        for cls in all_cls
    }

    # Step A: base = min(per_class_quota, pool_size)
    quota = {cls: min(per_class_quota, pool_size[cls]) for cls in all_cls}
    base_sum = sum(quota.values())

    if base_sum > target_n:
        # 罕见：per_class_quota * num_cls < target_n 但所有类都达不到 quota
        # 此时按 pool_size 比例下调（不会在我们当前数据上发生，写防御）
        raise RuntimeError(
            f"Step A 已分配 {base_sum} > target_n={target_n}，请调小 --per-class-quota"
        )

    remaining = target_n - base_sum
    # Step B: 给『还有余量』的类按余量比例分配剩余名额（最大余数法）
    if remaining > 0:
        rooms = []
        for cls in all_cls:
            room = pool_size[cls] - quota[cls]
            if room > 0:
                rooms.append((cls, room))
        total_room = sum(r for _, r in rooms)
        if total_room < remaining:
            raise RuntimeError(
                f"无法分配 target_n={target_n}：所有类总池只有 "
                f"{base_sum + total_room} 个样本"
            )
        # 计算每类应得的浮点份额，先取 floor，再按余数排序补齐
        raw_share = [(cls, remaining * r / total_room) for cls, r in rooms]
        int_share = {cls: int(s) for cls, s in raw_share}
        leftover = remaining - sum(int_share.values())
        # 余数从大到小给 +1
        sorted_by_frac = sorted(raw_share, key=lambda x: -(x[1] - int(x[1])))
        for cls, _ in sorted_by_frac[:leftover]:
            int_share[cls] += 1
        for cls, add in int_share.items():
            quota[cls] += add

    assert sum(quota.values()) == target_n, (
        f"quota allocation 失败 sum={sum(quota.values())} target={target_n}"
    )
    return quota


# --------------------------------------------------------------------------- #
# 给每类决定 (k_pos, k_neg)，使全局成功数严格命中 target_pos
# --------------------------------------------------------------------------- #

def assign_pos_per_class(
    quota: Dict[str, int],
    pool_pos: Dict[str, List[Dict[str, Any]]],
    pool_neg: Dict[str, List[Dict[str, Any]]],
    target_pos: int,
    rng: random.Random,
) -> Dict[str, int]:
    """
    返回 {class: k_pos}，满足：
      - 每类 k_pos ∈ [min_pos_c, max_pos_c]
      - sum(k_pos) == target_pos
      - k_neg_c = quota[c] - k_pos_c
    """
    classes = sorted(quota.keys())
    min_pos = {}
    max_pos = {}
    for c in classes:
        npos = len(pool_pos.get(c, []))
        nneg = len(pool_neg.get(c, []))
        q = quota[c]
        min_pos[c] = max(0, q - nneg)
        max_pos[c] = min(npos, q)
        if min_pos[c] > max_pos[c]:
            raise RuntimeError(
                f"class={c} 区间无效: min_pos={min_pos[c]} max_pos={max_pos[c]} "
                f"quota={q} pool=({npos}/{nneg})"
            )

    sum_min = sum(min_pos.values())
    sum_max = sum(max_pos.values())
    if not (sum_min <= target_pos <= sum_max):
        raise RuntimeError(
            f"target_pos={target_pos} 不在可行区间 [{sum_min}, {sum_max}]"
        )

    # 起始：每类先填 min_pos
    k_pos = dict(min_pos)
    deficit = target_pos - sum_min  # 还差多少 pos，要从下面这些类里 +1

    # 在每类的 (max_pos - k_pos) 余量里平均涨水位：
    # 用 round-robin，但走访顺序由 rng.shuffle 决定（不同 seed 出不同子集）。
    eligible = [c for c in classes if k_pos[c] < max_pos[c]]
    rng.shuffle(eligible)

    while deficit > 0 and eligible:
        next_eligible: List[str] = []
        for c in eligible:
            if deficit == 0:
                next_eligible.append(c)
                continue
            if k_pos[c] < max_pos[c]:
                k_pos[c] += 1
                deficit -= 1
                if k_pos[c] < max_pos[c]:
                    next_eligible.append(c)
        if not next_eligible:
            break
        eligible = next_eligible

    if deficit != 0:
        raise RuntimeError(
            f"涨水位算法异常：deficit={deficit} 无法清零（不应出现）"
        )

    assert sum(k_pos.values()) == target_pos
    return k_pos


# --------------------------------------------------------------------------- #
# 在每类的 pos / neg 池里实际抽样
# --------------------------------------------------------------------------- #

def pick_in_class(
    pool: List[Dict[str, Any]],
    k: int,
    iou_threshold: float,
    strategy: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    从一个类的 pos 池或 neg 池里挑 k 个样本。
    strategy:
        'representative' (默认): 按 |3d_iou - iou_threshold| 从大到小排序选取。
            - pos 池里就是选 IoU 最高的（最不会误判）
            - neg 池里就是选 IoU 最低的（最明显的失败）
        'random':            纯随机抽样（rng）
    """
    if k <= 0:
        return []
    if k >= len(pool):
        return list(pool)

    if strategy == "random":
        return rng.sample(pool, k)

    # representative: 距阈值越远越优先（用 -|x - thr| 排序，最远的最先）
    pool_sorted = sorted(
        pool, key=lambda s: -abs(float(s["3d_iou"]) - iou_threshold)
    )
    # 保证可复现：相同 IoU 的相对顺序由 rng 打乱
    # 先按 sort key 取前 k 个『紧贴 boundary 的兜底』里挑选
    return pool_sorted[:k]


# --------------------------------------------------------------------------- #
# 真正落盘
# --------------------------------------------------------------------------- #

def export_subset(
    chosen: List[Dict[str, Any]],
    src_dataset: Path,
    dst_dataset: Path,
) -> Dict[str, int]:
    """
    chosen: 已选样本列表，每个 dict 至少含 sample_id / class_name。
    把对应的 src_dataset/samples/<class>/<id>/ 整目录硬链接到 dst_dataset/samples/<class>/<id>/。
    返回 link 统计。
    """
    dst_samples_root = dst_dataset / "samples"
    ensure_dir(dst_samples_root)

    link_stats: Dict[str, int] = defaultdict(int)
    missing: List[str] = []

    for s in chosen:
        cls = s["class_name"]
        sid = s["sample_id"]
        src = src_dataset / "samples" / cls / sid
        dst = dst_samples_root / cls / sid
        if not src.exists():
            missing.append(f"{cls}/{sid}")
            link_stats["missing_in_src"] += 1
            continue
        if dst.exists():
            link_stats["skip_existing"] += 1
            continue
        ensure_dir(dst.parent)
        stats = hardlink_or_copy_dir(src, dst)
        for k, v in stats.items():
            link_stats[k] += v

    if missing:
        print(f"[subset] 警告：{len(missing)} 个样本在 src dataset 中找不到，已跳过：")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ...（还有 {len(missing) - 10} 个未列出）")

    return dict(link_stats)


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #

def render_report_html(report: Dict[str, Any]) -> str:
    totals = report["totals"]
    per_class = report["per_class"]
    rows = []
    for c in per_class:
        rows.append(
            f"<tr><td>{c['class_name']}</td><td>{c.get('object_prompt', '')}</td>"
            f"<td>{c['pool_pos']}/{c['pool_neg']}</td>"
            f"<td>{c['quota']}</td>"
            f"<td>{c['picked_pos']}/{c['picked_neg']}</td>"
            f"<td>{c['picked_acc']:.3f}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<title>Subset Dataset Report</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 1080px; margin: 24px auto; padding: 0 18px; color: #1f2937;
       line-height: 1.6; }}
h1 {{ color: #1e40af; border-bottom: 2px solid #1e40af; padding-bottom: 6px; }}
h2 {{ margin-top: 26px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }}
th {{ background: #f1f5f9; }}
.kv span {{ display: inline-block; min-width: 240px; }}
.note {{ background: #ecfeff; border-left: 4px solid #06b6d4;
         padding: 10px 14px; margin: 12px 0; }}
</style></head><body>

<h1>Subset Dataset Report</h1>
<div class="note">
  生成时间：{report['generated_at']}<br>
  源数据集：{report['src_dataset']}<br>
  目标数据集：{report['dst_dataset']}<br>
  评测结果：{report['results_file']}
</div>

<h2>1. 子集生成参数</h2>
<div class="kv">
  <span>IoU 阈值：<b>{report['iou_threshold']}</b></span>
  <span>目标样本数：<b>{report['target_n']}</b></span><br>
  <span>目标精度：<b>{report['target_acc']:.4f}</b></span>
  <span>目标 pos：<b>{report['target_pos']}</b></span>
  <span>目标 neg：<b>{report['target_neg']}</b></span><br>
  <span>每类 quota：<b>{report['per_class_quota']}</b></span>
  <span>采样策略：<b>{report['strategy']}</b></span>
  <span>RNG seed：<b>{report['seed']}</b></span>
</div>

<h2>2. 实际产出</h2>
<div class="kv">
  <span>实际样本数：<b>{totals['picked_n']}</b></span>
  <span>实际 pos：<b>{totals['picked_pos']}</b></span>
  <span>实际 neg：<b>{totals['picked_neg']}</b></span><br>
  <span>实际精度（IoU≥{report['iou_threshold']}）：
        <b>{totals['picked_acc']:.4f}</b></span>
  <span>覆盖类别数：<b>{totals['classes_covered']}</b></span>
</div>

<h2>3. 各类别详情</h2>
<table>
<tr><th>class</th><th>prompt</th><th>pool(p/n)</th><th>quota</th>
    <th>picked(p/n)</th><th>类内精度</th></tr>
{''.join(rows)}
</table>

</body></html>
"""
    return html


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="按目标精度从已发布数据集采样子集（finalize 之后用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-dataset", type=str, required=True,
                        help="源数据集目录（finalize_dataset.py 产物，含 samples/、samples.json 等）")
    parser.add_argument("--results", type=str, required=True,
                        help="benchmark 评测结果 JSON 文件（含 sample_results）")
    parser.add_argument("--dst-dataset", type=str, required=True,
                        help="目标子集数据集目录（不存在则创建）")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="判定『成功』的 3D IoU 阈值")
    parser.add_argument("--target-n", type=int, default=1000,
                        help="目标子集样本数（严格相等）")
    parser.add_argument("--target-acc", type=float, default=0.836,
                        help="目标精度（成功率），target_pos = round(target_n * target_acc)")
    parser.add_argument("--per-class-quota", type=int, default=25,
                        help="每类基础 quota（少于该数的类全收）")
    parser.add_argument(
        "--strategy", type=str, default="representative",
        choices=["representative", "random"],
        help="类内采样策略：representative=按 |IoU - 阈值| 远近选；random=纯随机",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（影响类别走访顺序与 random 策略）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只算配额、生成 report，不实际拷贝/链接")

    args = parser.parse_args()

    src_dataset = Path(args.src_dataset).expanduser().resolve()
    dst_dataset = Path(args.dst_dataset).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()

    if not src_dataset.exists():
        print(f"[subset] 源数据集不存在: {src_dataset}")
        return 1
    if not (src_dataset / "samples").exists():
        print(f"[subset] 源数据集缺 samples/ 子目录: {src_dataset}")
        return 1
    if not results_path.exists():
        print(f"[subset] results 文件不存在: {results_path}")
        return 1

    target_pos = round(args.target_n * args.target_acc)
    target_neg = args.target_n - target_pos
    actual_acc = target_pos / args.target_n
    print(f"[subset] src         = {src_dataset}")
    print(f"[subset] dst         = {dst_dataset}")
    print(f"[subset] results     = {results_path}")
    print(f"[subset] iou_thr     = {args.iou_threshold}")
    print(f"[subset] target_n    = {args.target_n}")
    print(f"[subset] target_acc  = {args.target_acc:.4f}  →  pos={target_pos} neg={target_neg}  (实际 acc={actual_acc:.4f})")
    print(f"[subset] strategy    = {args.strategy}")
    print(f"[subset] seed        = {args.seed}")
    if args.dry_run:
        print("[subset] dry-run mode: 不会写入 dst")

    rng = random.Random(args.seed)

    # 1) 读 results、拆 pool
    pool_pos, pool_neg = load_results_pool(results_path, args.iou_threshold)
    n_total = sum(len(v) for v in pool_pos.values()) + sum(len(v) for v in pool_neg.values())
    n_pos = sum(len(v) for v in pool_pos.values())
    n_neg = sum(len(v) for v in pool_neg.values())
    classes = sorted(set(pool_pos) | set(pool_neg))
    print(f"[subset] pool: {n_total} samples ({n_pos} pos / {n_neg} neg) "
          f"in {len(classes)} classes")

    if n_total < args.target_n:
        print(f"[subset] 错误：pool 总样本 {n_total} < target_n {args.target_n}")
        return 2
    if n_pos < target_pos:
        print(f"[subset] 错误：pool pos {n_pos} < target_pos {target_pos}")
        return 2
    if n_neg < target_neg:
        print(f"[subset] 错误：pool neg {n_neg} < target_neg {target_neg}")
        return 2

    # 2) 配额分配
    quota = allocate_quotas(pool_pos, pool_neg, args.target_n, args.per_class_quota)

    # 3) 给每类决定 k_pos / k_neg，让全局命中 target_pos
    k_pos = assign_pos_per_class(
        quota, pool_pos, pool_neg, target_pos, rng,
    )
    # k_neg = quota - k_pos
    k_neg = {c: quota[c] - k_pos[c] for c in quota}

    # 4) 类内抽样
    chosen: List[Dict[str, Any]] = []
    per_class_records: List[Dict[str, Any]] = []
    for cls in classes:
        pos_picks = pick_in_class(
            pool_pos.get(cls, []), k_pos[cls],
            args.iou_threshold, args.strategy, rng,
        )
        neg_picks = pick_in_class(
            pool_neg.get(cls, []), k_neg[cls],
            args.iou_threshold, args.strategy, rng,
        )
        picks = pos_picks + neg_picks
        chosen.extend(picks)

        prompt = ""
        if pool_pos.get(cls):
            prompt = pool_pos[cls][0].get("object_prompt", "")
        elif pool_neg.get(cls):
            prompt = pool_neg[cls][0].get("object_prompt", "")
        per_class_records.append({
            "class_name": cls,
            "object_prompt": prompt,
            "pool_pos": len(pool_pos.get(cls, [])),
            "pool_neg": len(pool_neg.get(cls, [])),
            "quota": quota[cls],
            "picked_pos": len(pos_picks),
            "picked_neg": len(neg_picks),
            "picked_acc": (
                len(pos_picks) / max(1, len(pos_picks) + len(neg_picks))
            ),
        })

    # 5) 全局 sanity 校验
    picked_n = len(chosen)
    picked_pos = sum(1 for s in chosen if float(s["3d_iou"]) >= args.iou_threshold)
    picked_neg = picked_n - picked_pos
    picked_acc = picked_pos / max(1, picked_n)

    print()
    print("=== 子集分配结果 ===")
    print(f"  total picked: {picked_n}  (target {args.target_n})")
    print(f"  picked pos  : {picked_pos}  (target {target_pos})")
    print(f"  picked neg  : {picked_neg}  (target {target_neg})")
    print(f"  picked acc  : {picked_acc:.4f}  (target {args.target_acc:.4f})")
    print(f"  classes     : {len(classes)} 全覆盖")
    if picked_n != args.target_n:
        print(f"[subset] ⚠ 总数不匹配 (差 {picked_n - args.target_n})")
    if picked_pos != target_pos:
        print(f"[subset] ⚠ 成功数不匹配 (差 {picked_pos - target_pos})")

    # 6) 写新数据集（如非 dry-run）
    src_index = load_src_index(src_dataset)
    samples_index_out: List[Dict[str, Any]] = []
    for s in chosen:
        sid = s["sample_id"]
        cls = s["class_name"]
        rec = {
            "sample_id": sid,
            "class_name": cls,
            "sample_dir": f"samples/{cls}/{sid}",
            "object_prompt": s.get("object_prompt", ""),
            "eval_3d_iou": float(s["3d_iou"]),
            "eval_is_correct_at_threshold": (
                float(s["3d_iou"]) >= args.iou_threshold
            ),
        }
        # 把原 dataset/samples.json 里的 class_id / class_name_zh 等带过来（best-effort）
        src_rec = src_index.get(sid)
        if isinstance(src_rec, dict):
            for k in ("class_id", "class_name_zh"):
                if k in src_rec:
                    rec[k] = src_rec[k]
        samples_index_out.append(rec)

    link_stats: Dict[str, int] = {}
    if not args.dry_run:
        ensure_dir(dst_dataset)
        link_stats = export_subset(chosen, src_dataset, dst_dataset)

        # samples.json
        save_json(dst_dataset / "samples.json", samples_index_out)

        # classes.json：只保留出现过的类，从 src 继承字段
        src_classes = []
        src_classes_path = src_dataset / "classes.json"
        if src_classes_path.exists():
            src_classes_doc = load_json(src_classes_path, default=None)
            if isinstance(src_classes_doc, list):
                src_classes = src_classes_doc
            elif isinstance(src_classes_doc, dict) and isinstance(
                src_classes_doc.get("classes"), list
            ):
                src_classes = src_classes_doc["classes"]
        picked_class_set = set(classes)
        kept_classes = [
            c for c in src_classes
            if isinstance(c, dict) and c.get("name") in picked_class_set
        ]
        save_json(dst_dataset / "classes.json", kept_classes)

        # dataset.json：全局元数据
        save_json(dst_dataset / "dataset.json", {
            "name": dst_dataset.name,
            "generated_at": now_iso(),
            "generator": "build_subset_dataset.py",
            "src_dataset": str(src_dataset),
            "results_file": str(results_path),
            "iou_threshold": args.iou_threshold,
            "target_n": args.target_n,
            "target_acc": args.target_acc,
            "actual_n": picked_n,
            "actual_pos": picked_pos,
            "actual_neg": picked_neg,
            "actual_acc": picked_acc,
            "classes_covered": len(classes),
            "per_class_quota_base": args.per_class_quota,
            "strategy": args.strategy,
            "seed": args.seed,
            "link_stats": link_stats,
        })

    # 7) 报告
    report = {
        "generated_at": now_iso(),
        "src_dataset": str(src_dataset),
        "dst_dataset": str(dst_dataset),
        "results_file": str(results_path),
        "iou_threshold": args.iou_threshold,
        "target_n": args.target_n,
        "target_acc": args.target_acc,
        "target_pos": target_pos,
        "target_neg": target_neg,
        "per_class_quota": args.per_class_quota,
        "strategy": args.strategy,
        "seed": args.seed,
        "totals": {
            "picked_n": picked_n,
            "picked_pos": picked_pos,
            "picked_neg": picked_neg,
            "picked_acc": picked_acc,
            "classes_covered": len(classes),
        },
        "per_class": sorted(per_class_records, key=lambda x: x["class_name"]),
        "link_stats": link_stats,
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        save_json(dst_dataset / "subset_meta.json", report)
        (dst_dataset / "subset_report.html").write_text(
            render_report_html(report), encoding="utf-8",
        )
        print()
        print(f"[subset] ✓ 子集已写入: {dst_dataset}")
        print(f"  link_stats: {link_stats}")
        print(f"  报告:       {dst_dataset / 'subset_report.html'}")
    else:
        print()
        print("[subset] dry-run 完毕，未写入 dst")
        print(f"  per_class 详情见 stdout（如需 HTML 报告请去掉 --dry-run 重跑）")
        for r in sorted(per_class_records, key=lambda x: x["class_name"]):
            print(
                f"  {r['class_name']:35s}({r['object_prompt']:6s})  "
                f"pool={r['pool_pos']:3d}/{r['pool_neg']:3d}  "
                f"quota={r['quota']:3d}  picked={r['picked_pos']:3d}/{r['picked_neg']:3d}  "
                f"acc={r['picked_acc']:.3f}"
            )

    return 0 if (picked_n == args.target_n and picked_pos == target_pos) else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[subset][fatal] {exc}")
        traceback.print_exc()
        sys.exit(2)
