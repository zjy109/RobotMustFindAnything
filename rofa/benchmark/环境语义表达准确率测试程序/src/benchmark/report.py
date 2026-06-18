"""HTML 报告生成器（纯字符串 + SVG，无 matplotlib 依赖）。

结构：
1. 顶部：总精度 / 成功数 / 总样本数 / IoU 阈值
2. 类别精度柱状图（SVG）
3. Top-30 失败样本缩略图（按 IoU 升序）+ 其余失败样本 ID 列表
4. 环境快照
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BenchmarkConfig


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _encode_image_b64(path: Path, max_kb: int = 200) -> Optional[str]:
    """读取图片并 base64 编码。文件过大或不存在时返回 None。"""
    if not path.exists():
        return None
    try:
        size = path.stat().st_size
        if size > max_kb * 1024:
            # 太大就不嵌入了，留外链
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None


def _all_samples(results_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cls_block in (results_doc.get("sample_results") or {}).values():
        if isinstance(cls_block, dict):
            out.extend(cls_block.get("samples", []))
    return out


# ---------------------------------------------------------------------------
# SVG 柱状图（类别精度）
# ---------------------------------------------------------------------------

def render_class_accuracy_svg(per_class: List[Dict[str, Any]]) -> str:
    """画一个 SVG 横向柱状图，类名在左，柱子在右。"""
    if not per_class:
        return "<p>(无类别数据)</p>"

    bar_h = 18
    pad_top = 10
    pad_bottom = 10
    label_w = 140
    bar_max_w = 360
    n = len(per_class)
    height = pad_top + pad_bottom + n * (bar_h + 4)
    width = label_w + bar_max_w + 80

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family: -apple-system, sans-serif; font-size: 12px;">']
    # 网格线（每 0.2 一条）
    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        x = label_w + bar_max_w * v
        svg.append(
            f'<line x1="{x}" y1="{pad_top - 2}" x2="{x}" y2="{height - pad_bottom + 2}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x}" y="{height - 2}" text-anchor="middle" fill="#6b7280">{v:.1f}</text>'
        )

    sorted_pc = sorted(per_class, key=lambda c: -c.get("accuracy_3d", 0))
    for i, c in enumerate(sorted_pc):
        y = pad_top + i * (bar_h + 4)
        acc = float(c.get("accuracy_3d", 0))
        bar_w = bar_max_w * max(0.0, min(1.0, acc))
        # 标签
        name = html.escape(str(c.get("class_zh") or c.get("class_name") or ""))[:14]
        svg.append(
            f'<text x="{label_w - 4}" y="{y + bar_h - 4}" text-anchor="end" '
            f'fill="#1f2937">{name}</text>'
        )
        # 柱
        color = "#10b981" if acc >= 0.8 else ("#f59e0b" if acc >= 0.5 else "#ef4444")
        svg.append(
            f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" '
            f'fill="{color}" rx="2"/>'
        )
        # 数值
        svg.append(
            f'<text x="{label_w + bar_w + 4}" y="{y + bar_h - 4}" '
            f'fill="#374151">{acc:.3f} ({c.get("total_samples", 0)})</text>'
        )
    svg.append("</svg>")
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# 失败案例展板
# ---------------------------------------------------------------------------

def render_failure_gallery(
    samples: List[Dict[str, Any]],
    viz_dir: Path,
    top_k: int = 30,
) -> str:
    """选 IoU 最低的 K 个失败样本，做一个缩略图廊。"""
    failures = [s for s in samples if not s.get("is_correct")]
    failures.sort(key=lambda s: float(s.get("3d_iou", 0)))
    top = failures[:top_k]
    rest = failures[top_k:]

    if not failures:
        return '<p style="color:#10b981">没有失败样本，恭喜！</p>'

    cards = []
    for s in top:
        sid = s.get("sample_id", "?")
        iou = float(s.get("3d_iou", 0))
        cls = html.escape(str(s.get("object_prompt", "")))
        viz_path = viz_dir / f"det_seg_{sid}.jpg"
        b64 = _encode_image_b64(viz_path, max_kb=300)
        if b64:
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;border-radius:6px"/>'
        else:
            img_tag = (
                f'<div style="height:140px;background:#f3f4f6;border-radius:6px;'
                f'display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:12px">'
                f'(无可视化图)</div>'
            )
        cards.append(
            f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:8px;background:#fff">'
            f'{img_tag}'
            f'<div style="font-size:11px;color:#374151;margin-top:6px">'
            f'<b>{html.escape(sid)}</b> · {cls}</div>'
            f'<div style="font-size:11px;color:#dc2626">3D IoU = {iou:.4f}</div>'
            f'</div>'
        )

    gallery = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));'
        f'gap:10px;margin-top:10px">{"".join(cards)}</div>'
    )

    rest_block = ""
    if rest:
        rest_ids = [html.escape(s.get("sample_id", "?")) for s in rest]
        rest_block = (
            f'<details style="margin-top:14px"><summary>'
            f'其余 {len(rest)} 个失败样本 ID（点击展开）</summary>'
            f'<div style="font-family:monospace;font-size:12px;color:#374151;margin-top:8px">'
            f'{", ".join(rest_ids)}</div></details>'
        )

    return gallery + rest_block


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def render_html_report(
    cfg: BenchmarkConfig, results_doc: Dict[str, Any], env_snap: Dict[str, Any],
) -> str:
    summary = results_doc.get("summary", {})
    sample_results = results_doc.get("sample_results", {})

    # 类别表 & 总样本
    per_class: List[Dict[str, Any]] = []
    for cls_zh, block in sample_results.items():
        if not isinstance(block, dict):
            continue
        st = block.get("stats", {})
        per_class.append({
            "class_zh": cls_zh,
            "total_samples": st.get("total_samples", 0),
            "avg_2d_iou": st.get("avg_2d_iou", 0),
            "avg_3d_iou": st.get("avg_3d_iou", 0),
            "accuracy_3d": st.get("accuracy_3d", 0),
        })

    samples = _all_samples(results_doc)

    rows_class_html = "".join(
        f"<tr><td>{html.escape(str(c['class_zh']))}</td>"
        f"<td>{c['total_samples']}</td>"
        f"<td>{c['avg_2d_iou']:.4f}</td>"
        f"<td>{c['avg_3d_iou']:.4f}</td>"
        f"<td><b>{c['accuracy_3d']:.4f}</b></td></tr>"
        for c in sorted(per_class, key=lambda x: -x["accuracy_3d"])
    )

    bar_svg = render_class_accuracy_svg(per_class)
    failure_html = render_failure_gallery(
        samples, cfg.visualizations_dir, top_k=30,
    )

    # 环境
    env_html = "<pre style='font-size:11px;background:#f3f4f6;padding:10px;border-radius:6px;overflow-x:auto'>"
    env_html += html.escape(json.dumps(env_snap, ensure_ascii=False, indent=2)[:4000])
    env_html += "</pre>"

    overall_acc = float(summary.get("accuracy", 0))
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<title>环境语义表达准确率测试报告</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 1200px; margin: 24px auto; padding: 0 18px; color: #1f2937;
       line-height: 1.6; background: #fafafa; }}
h1 {{ color: #1e40af; border-bottom: 3px solid #1e40af; padding-bottom: 8px; }}
h2 {{ margin-top: 32px; padding-left: 6px; border-left: 4px solid #1e40af; }}
.kv {{ display: flex; flex-wrap: wrap; gap: 14px 32px; padding: 14px;
       background: #fff; border-radius: 8px; margin-top: 8px;
       border: 1px solid #e5e7eb; }}
.kv .item b {{ color: #1e40af; font-size: 16px; }}
.note {{ background: #ecfeff; border-left: 4px solid #06b6d4;
         padding: 10px 14px; margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }}
th {{ background: #f1f5f9; }}
.metric-card {{ display: inline-block; padding: 12px 18px; border-radius: 8px;
                background: #fff; margin-right: 12px; min-width: 160px;
                border: 1px solid #e5e7eb; }}
.metric-card .v {{ font-size: 26px; font-weight: 700; color: #1e40af; }}
.metric-card .l {{ font-size: 12px; color: #6b7280; }}
</style></head><body>

<h1>环境语义表达准确率测试报告</h1>
<div class="note">
  生成时间：{html.escape(env_snap.get("generated_at", ""))}<br>
  数据集：<code>{html.escape(str(cfg.dataset_root))}</code><br>
  输出目录：<code>{html.escape(str(cfg.output_dir))}</code>
</div>

<h2>1. 核心指标</h2>
<div>
  <div class="metric-card"><div class="v">{overall_acc:.4f}</div>
    <div class="l">总精度（IoU≥{cfg.iou_threshold}）</div></div>
  <div class="metric-card"><div class="v">{summary.get("success_count", 0)}</div>
    <div class="l">成功样本数</div></div>
  <div class="metric-card"><div class="v">{summary.get("total_processed", 0)}</div>
    <div class="l">已处理样本数</div></div>
  <div class="metric-card"><div class="v">{len(per_class)}</div>
    <div class="l">类别覆盖</div></div>
</div>
<div class="note" style="margin-top:14px;background:#fef3c7;border-left-color:#d97706">
  <b>锁死的核心评测参数</b>（与课题二原 <code>benchmark.py</code> 一致）：
  IoU 阈值=<code>{cfg.iou_threshold}</code>　
  SOR=<code>nb={cfg.sor_nb}, std={cfg.sor_std}</code>　
  存在性预筛=<code>{cfg.enable_existence_check}</code>　
  SAM2=<code>{html.escape(cfg.sam2_model_id)}</code>
</div>

<h2>2. 各类别精度</h2>
<div style="margin-top:10px;background:#fff;padding:14px;border-radius:8px;border:1px solid #e5e7eb;overflow-x:auto">
{bar_svg}
</div>

<details style="margin-top:14px"><summary>查看类别精度数值表（{len(per_class)} 类）</summary>
<table style="margin-top:10px">
<tr><th>类别</th><th>样本数</th><th>2D IoU 均值</th><th>3D IoU 均值</th><th>精度</th></tr>
{rows_class_html}
</table>
</details>

<h2>3. 失败案例（Top-30 by 3D IoU 升序）</h2>
{failure_html}

<h2>4. 运行环境快照</h2>
{env_html}

<hr style="margin-top:40px;border:none;border-top:1px solid #e5e7eb">
<p style="color:#6b7280;font-size:12px;text-align:center">
  环境语义表达准确率测试程序 · 报告自动生成
</p>

</body></html>
"""


def write_html_report(
    cfg: BenchmarkConfig, results_doc: Dict[str, Any], env_snap: Dict[str, Any],
) -> Path:
    html_text = render_html_report(cfg, results_doc, env_snap)
    cfg.report_html.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_html.write_text(html_text, encoding="utf-8")
    return cfg.report_html


def write_json_report(
    cfg: BenchmarkConfig, results_doc: Dict[str, Any],
) -> Path:
    """更精炼的结果摘要，只含汇总数字 + 每类精度。"""
    summary = results_doc.get("summary", {})
    sample_results = results_doc.get("sample_results", {})
    per_class = []
    for cls_zh, block in sample_results.items():
        if not isinstance(block, dict):
            continue
        st = block.get("stats", {})
        per_class.append({"class_zh": cls_zh, **st})

    doc = {
        "summary": summary,
        "iou_threshold": float(cfg.iou_threshold),
        "per_class": sorted(per_class, key=lambda x: -x.get("accuracy_3d", 0)),
    }
    cfg.report_json.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return cfg.report_json
