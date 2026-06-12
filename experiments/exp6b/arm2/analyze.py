"""Exp-6b ARM 2, Phase D: analysis of multi-judge detection surfaces.

Evaluates exactly the pre-registered quantities (docs/exp6b_arm2_prereg.md):
per-judge detection surface, FPR on delta=0, malformed rate, boundary
estimates per (onset, shape), granularity check, and P1-P3. Ground truth
(distance_to_correct) is read from the Phase A CSVs here, in analysis only.

    python -m experiments.exp6b.arm2.analyze

Writes results/exp6b_arm2/analysis.json and docs/experiment_6b_arm2_results.md.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.exp6b.arm2.gridgen import DELTAS, MANIFEST_PATH, ONSETS, SHAPES, TRACES_DIR
from experiments.exp6b.arm2.judge_harness import (
    JUDGES,
    JUDGMENTS_DIR,
    N_STEPS,
    SUBSAMPLE_SEEDS,
    judged_steps_main,
)

ANALYSIS_PATH = _REPO_ROOT / "results" / "exp6b_arm2" / "analysis.json"
DOC_PATH = _REPO_ROOT / "docs" / "experiment_6b_arm2_results.md"

FIRE_RATE_BOUNDARY = 0.70   # prereg §7 P3
FPR_LIMIT = 0.05            # prereg §7 P2
MALFORMED_LIMIT = 0.10      # prereg §4


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_manifest_cells() -> dict[str, dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {c["cell"]: c for c in manifest["cells"]}


def load_judgments(tag: str) -> list[dict[str, Any]]:
    path = JUDGMENTS_DIR / f"{tag}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def load_trace_distances() -> dict[str, dict[int, float]]:
    """trace_id -> {step: distance_to_correct} from Phase A CSVs (ground truth)."""
    out: dict[str, dict[int, float]] = {}
    for csv_path in TRACES_DIR.glob("exp6b2_*.csv"):
        trace_id = csv_path.stem.removeprefix("exp6b2_")
        with csv_path.open(encoding="utf-8") as fh:
            out[trace_id] = {
                int(row["step"]): float(row["distance_to_correct"])
                for row in csv.DictReader(fh)
            }
    return out


# ---------------------------------------------------------------------------
# Fire rule (mirrors prereg §6)
# ---------------------------------------------------------------------------

def fires(judged_seq: list[tuple[int, str]], onset: int) -> bool:
    """>=2 consecutive advancing='no' at judged steps >= onset."""
    run = 0
    for step, advancing in sorted(judged_seq):
        if step < onset:
            continue
        run = run + 1 if advancing == "no" else 0
        if run >= 2:
            return True
    return False


def cell_onset(cell_meta: dict[str, Any]) -> int:
    return cell_meta["onset"] if cell_meta["onset"] is not None else 1


# ---------------------------------------------------------------------------
# Correlation helpers (no external deps)
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


# ---------------------------------------------------------------------------
# Per-judge analysis
# ---------------------------------------------------------------------------

def analyze_judge(
    tag: str,
    cells: dict[str, dict[str, Any]],
    distances: dict[str, dict[int, float]],
) -> dict[str, Any]:
    judgments = load_judgments(tag)
    if not judgments:
        return {"judge": tag, "status": "NO DATA"}

    n_malformed = sum(1 for j in judgments if j["malformed"])
    malformed_rate = n_malformed / len(judgments)
    unusable = malformed_rate > MALFORMED_LIMIT

    main = [j for j in judgments if j["pass"] == "main"]
    by_trace_main: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for j in main:
        by_trace_main[j["trace_id"]].append((j["step"], j["advancing"]))

    # Detection surface: fire rate per cell
    cell_fire: dict[str, dict[str, Any]] = {}
    for cell_name, meta in cells.items():
        onset = cell_onset(meta)
        trace_ids = [t for t in by_trace_main if t.rsplit("_seed", 1)[0] == cell_name]
        fired = [t for t in trace_ids if fires(by_trace_main[t], onset)]
        cell_fire[cell_name] = {
            "n": len(trace_ids),
            "n_fired": len(fired),
            "fire_rate": round(len(fired) / len(trace_ids), 3) if trace_ids else None,
        }

    fpr = cell_fire.get("d0", {}).get("fire_rate")

    # Pooled fire rate per delta (P1) — nonzero cells pooled over onset/shape
    pooled_by_delta: dict[float, list[int]] = {}
    for cell_name, meta in cells.items():
        if meta["delta"] == 0.0:
            continue
        s = pooled_by_delta.setdefault(meta["delta"], [0, 0])
        s[0] += cell_fire[cell_name]["n_fired"]
        s[1] += cell_fire[cell_name]["n"]
    pooled_rates = {
        d: round(s[0] / s[1], 3) if s[1] else None
        for d, s in sorted(pooled_by_delta.items())
    }
    rates_seq = [pooled_rates[d] for d in sorted(pooled_rates)]
    p1_monotone = all(
        a is not None and b is not None and a <= b
        for a, b in zip(rates_seq, rates_seq[1:])
    ) if rates_seq else False

    # Boundary per (onset, shape): smallest delta with fire rate >= 0.70
    boundaries: dict[str, float | None] = {}
    for onset in ONSETS:
        for shape in SHAPES:
            boundary = None
            for delta in sorted(DELTAS):
                cn = f"d{delta:g}_o{onset}_{shape}"
                rate = cell_fire.get(cn, {}).get("fire_rate")
                if rate is not None and rate >= FIRE_RATE_BOUNDARY:
                    boundary = delta
                    break
            boundaries[f"o{onset}_{shape}"] = boundary

    # Granularity check: per-step fire bit vs main fire bit on subsample traces
    perstep = [j for j in judgments if j["pass"] == "perstep"]
    by_trace_all: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for j in main + perstep:
        by_trace_all[j["trace_id"]].append((j["step"], j["advancing"]))
    sub_ids = sorted({j["trace_id"] for j in perstep})
    agree = disagree = 0
    disagreements: list[dict[str, Any]] = []
    for t in sub_ids:
        cell_name = t.rsplit("_seed", 1)[0]
        onset = cell_onset(cells[cell_name])
        f_main = fires(by_trace_main[t], onset)
        f_full = fires(by_trace_all[t], onset)
        if f_main == f_full:
            agree += 1
        else:
            disagree += 1
            disagreements.append({"trace_id": t, "main": f_main, "perstep": f_full})

    # Ground-truth correlation (analysis only): progress vs distance_to_correct
    xs, ys = [], []
    for j in main:
        if j["malformed"] or j["progress"] is None:
            continue
        d = distances.get(j["trace_id"], {}).get(j["step"])
        if d is not None:
            xs.append(j["progress"])
            ys.append(d)
    pearson = _pearson(xs, ys)
    spearman = _spearman(xs, ys)

    return {
        "judge": tag,
        "n_judgments": len(judgments),
        "n_main": len(main),
        "malformed_rate": round(malformed_rate, 4),
        "unusable": unusable,
        "fpr_delta0": fpr,
        "cell_fire": cell_fire,
        "pooled_fire_by_delta": pooled_rates,
        "p1_monotone": p1_monotone,
        "p2_fpr_ok": (fpr is not None and fpr <= FPR_LIMIT),
        "boundaries": boundaries,
        "granularity": {
            "n_subsample": len(sub_ids), "agree": agree, "disagree": disagree,
            "disagreements": disagreements,
        },
        "gt_correlation": {
            "n": len(xs),
            "pearson_progress_vs_distance": round(pearson, 4) if pearson is not None else None,
            "spearman_progress_vs_distance": round(spearman, 4) if spearman is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# Cross-judge P3
# ---------------------------------------------------------------------------

def evaluate_p3(per_judge: list[dict[str, Any]]) -> dict[str, Any]:
    """Boundaries should shift toward smaller delta as scale increases."""
    usable = [r for r in per_judge if not r.get("unusable") and "boundaries" in r]
    # JUDGES order is smallest-first; preserve it
    order = [j["tag"] for j in JUDGES]
    usable.sort(key=lambda r: order.index(r["judge"]))

    groups: dict[str, list[Any]] = {}
    for onset in ONSETS:
        for shape in SHAPES:
            key = f"o{onset}_{shape}"
            groups[key] = [r["boundaries"].get(key) for r in usable]

    def _shifts_smaller(seq: list[Any]) -> str:
        # None (never reaches 70%) is treated as a boundary beyond the largest delta
        big = max(DELTAS) * 10
        vals = [big if v is None else v for v in seq]
        if all(a >= b for a, b in zip(vals, vals[1:])) and vals[0] > vals[-1]:
            return "shifts_smaller"
        if all(a == b for a, b in zip(vals, vals[1:])):
            return "flat"
        if all(a >= b for a, b in zip(vals, vals[1:])):
            return "non-increasing (weak)"
        return "non-monotone"

    verdicts = {k: _shifts_smaller(v) for k, v in groups.items()}
    n_shift = sum(1 for v in verdicts.values() if v == "shifts_smaller")
    n_flat = sum(1 for v in verdicts.values() if v == "flat")
    return {
        "judge_order": [r["judge"] for r in usable],
        "boundaries_by_group": groups,
        "group_verdicts": verdicts,
        "n_groups_shifting_smaller": n_shift,
        "n_groups_flat": n_flat,
        "boundaries_move": n_shift > 0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(per_judge: list[dict[str, Any]], p3: dict[str, Any],
                 cells: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Exp-6b ARM 2 — Phase D Results: Oracle Detection Surfaces, Multi-Judge",
        "",
        "**Date:** 2026-06-11  **Pre-registration:** `docs/exp6b_arm2_prereg.md` "
        "(frozen before any judge call)  **Grid:** `docs/exp6b_arm2_grid.md`",
        "",
        "All quantities below are evaluated exactly as pre-registered; anything "
        "outside P1-P3 is labeled exploratory. Ground truth "
        "(`distance_to_correct`) entered analysis only, never any judge prompt.",
        "",
    ]

    # Per-judge sections
    for r in per_judge:
        lines.append(f"## Judge: {r['judge']}")
        lines.append("")
        if r.get("status") == "NO DATA":
            lines += ["**NO DATA** — judge did not run.", ""]
            continue
        unusable_note = (
            " — **UNUSABLE per prereg §4 (>10%); results below are reported "
            "for completeness but excluded from P3**" if r["unusable"] else ""
        )
        lines += [
            f"- Judgments: {r['n_judgments']} ({r['n_main']} main-pass)",
            f"- Malformed-output rate: **{r['malformed_rate']:.2%}**{unusable_note}",
            f"- FPR on delta=0 controls: **{r['fpr_delta0']}** "
            f"(P2 ≤ 5%: {'PASS' if r['p2_fpr_ok'] else '**FAIL**'})",
            f"- Pooled fire rate by delta: "
            + ", ".join(f"d{d:g}={v}" for d, v in r["pooled_fire_by_delta"].items())
            + f"  (P1 monotone: {'PASS' if r['p1_monotone'] else '**FAIL**'})",
            f"- Granularity check (prereg §5): {r['granularity']['agree']}/"
            f"{r['granularity']['n_subsample']} fire decisions agree between "
            f"every-3rd-step and per-step judging"
            + (f"; disagreements: {r['granularity']['disagreements']}"
               if r["granularity"]["disagree"] else ""),
            f"- Ground-truth correlation (n={r['gt_correlation']['n']}): "
            f"Pearson(progress, distance) = "
            f"{r['gt_correlation']['pearson_progress_vs_distance']}, "
            f"Spearman = {r['gt_correlation']['spearman_progress_vs_distance']} "
            "(negative = progress tracks truth)",
            "",
            "### Detection surface (fire rate per cell)",
            "",
            "| delta | onset | shape | fire rate (n fired / n) |",
            "|---|---|---|---|",
        ]
        for cell_name, meta in sorted(
            cells.items(), key=lambda kv: (kv[1]["delta"], kv[1]["onset"] or 0,
                                           kv[1]["shape"] or "")
        ):
            cf = r["cell_fire"].get(cell_name, {})
            onset = meta["onset"] if meta["onset"] is not None else "—"
            shape = meta["shape"] if meta["shape"] is not None else "—"
            lines.append(
                f"| {meta['delta']:g} | {onset} | {shape} "
                f"| {cf.get('fire_rate')} ({cf.get('n_fired')}/{cf.get('n')}) |"
            )
        lines += [
            "",
            "### Boundary estimates (smallest delta with ≥70% fire rate)",
            "",
            "| onset/shape | boundary delta |",
            "|---|---|",
        ]
        for k, v in r["boundaries"].items():
            lines.append(f"| {k} | {v if v is not None else 'none (never ≥70%)'} |")
        lines.append("")

    # Cross-judge
    lines += [
        "## Cross-judge: P1-P3 verdicts (pre-registered)",
        "",
        "### P1 — fire rate increases monotonically with delta, per judge",
        "",
    ]
    for r in per_judge:
        if "p1_monotone" in r:
            lines.append(f"- {r['judge']}: {'**PASS**' if r['p1_monotone'] else '**FAIL**'} "
                         f"({', '.join(f'd{d:g}={v}' for d, v in r['pooled_fire_by_delta'].items())})")
    lines += [
        "",
        "### P2 — FPR on delta=0 ≤ 5%, per judge",
        "",
    ]
    for r in per_judge:
        if "p2_fpr_ok" in r:
            lines.append(f"- {r['judge']}: FPR={r['fpr_delta0']} — "
                         f"{'**PASS**' if r['p2_fpr_ok'] else '**FAIL**'}")
    lines += [
        "",
        "### P3 — detection boundary shifts toward smaller delta with judge scale "
        "(LOAD-BEARING)",
        "",
        f"Judge order (total params, smallest first): {p3['judge_order']}",
        "",
        "| onset/shape | " + " | ".join(p3["judge_order"]) + " | verdict |",
        "|---|" + "---|" * (len(p3["judge_order"]) + 1),
    ]
    for k, seq in p3["boundaries_by_group"].items():
        cells_str = " | ".join("none" if v is None else f"{v:g}" for v in seq)
        lines.append(f"| {k} | {cells_str} | {p3['group_verdicts'][k]} |")
    lines += [
        "",
        f"Groups where the boundary shifts smaller with scale: "
        f"**{p3['n_groups_shifting_smaller']} / {len(p3['boundaries_by_group'])}** "
        f"(flat: {p3['n_groups_flat']}).",
        "",
    ]
    if not p3["boundaries_move"]:
        lines += [
            "**FLAG (pre-registered): boundaries do NOT move across local scales.** "
            "Per prereg §7, this predicts that a frontier judge may not rescue the "
            "oracle channel — the failure is in the channel, not merely in judge "
            "capability.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    cells = load_manifest_cells()
    distances = load_trace_distances()
    per_judge = [analyze_judge(j["tag"], cells, distances) for j in JUDGES]
    p3 = evaluate_p3(per_judge)

    ANALYSIS_PATH.write_text(
        json.dumps({"per_judge": per_judge, "p3": p3}, indent=2), encoding="utf-8"
    )
    print(f"Analysis JSON -> {ANALYSIS_PATH.relative_to(_REPO_ROOT)}")

    report = build_report(per_judge, p3, cells)
    DOC_PATH.write_text(report, encoding="utf-8")
    print(f"Report -> {DOC_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
