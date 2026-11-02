#!/usr/bin/env python3
"""Recompute the frozen numerical tables and verify this source release."""

from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"add": 115, "remove": 149, "replace": 79, "volume": 77}
CELLS = ["A0B0", "A1B0", "A0B1", "A1B1"]
METRICS = {"IFR": "instruction_following", "CR": "consistency", "EMR": "exact_match"}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")


def compute():
    cohort_ids = {}
    t4 = []
    raw = {}
    for task, n in TASKS.items():
        for cell in CELLS:
            rows = read(ROOT / "results" / cell / task / "per_sample.json")
            if len(rows) != n or len({r["sample_id"] for r in rows}) != n:
                raise ValueError("Incomplete or duplicate cohort")
            if any(
                not isinstance(r[k], (int, float)) or not math.isfinite(r[k]) or not 0 <= r[k] <= 1
                for r in rows
                for k in METRICS.values()
            ):
                raise ValueError("Invalid score")
            ids = {r["sample_id"] for r in rows}
            if task in cohort_ids and ids != cohort_ids[task]:
                raise ValueError("Different cohort across configurations")
            cohort_ids[task] = ids
            raw[cell, task] = rows
            t4.append(
                dict(
                    task=task,
                    cell=cell,
                    n=n,
                    **{m: 100 * math.fsum(r[k] for r in rows) / n for m, k in METRICS.items()},
                    all_pass_count=sum(r["exact_match"] == 1 for r in rows),
                )
            )
    t5 = []
    for cell in CELLS:
        rows = [r for r in t4 if r["cell"] == cell]
        pooled = [r for task in TASKS for r in raw[cell, task]]
        t5.append(
            dict(
                cell=cell,
                **{"macro_" + m: math.fsum(r[m] for r in rows) / 4 for m in METRICS},
                **{
                    "pooled_" + m: 100 * math.fsum(r[k] for r in pooled) / len(pooled)
                    for m, k in METRICS.items()
                },
                all_pass_count=sum(r["exact_match"] == 1 for r in pooled),
            )
        )
    expected = read(ROOT / "configs/expected_tables.json")
    checks = 0
    for actual, wanted, keys in [
        (t4, expected["current_table4"], list(METRICS)),
        (
            t5,
            expected["current_table5"],
            ["macro_" + m for m in METRICS] + ["pooled_" + m for m in METRICS],
        ),
    ]:
        for a, e in zip(actual, wanted, strict=True):
            if a["cell"] != e["cell"] or a.get("task") != e.get("task"):
                raise ValueError("Wrong version ordering")
            for k in keys:
                if f"{a[k]:.2f}" != f"{e[k]:.2f}":
                    raise ValueError(f"Paper value mismatch {a.get('task')} {a['cell']} {k}")
                checks += 1
    return t4, t5, checks


def tables(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t4, t5, checks = compute()
    for name, rows in [
        ("table4", t4),
        ("table5", t5),
        ("main", [r for r in t4 if r["cell"] == "A1B1"]),
    ]:
        with (out / (name + ".csv")).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write(out / "tables.json", dict(table4=t4, table5=t5, paper_metric_checks=checks))

    def render(r, k, rows):
        s = f"{r[k]:.2f}"
        return "\\textbf{" + s + "}" if r[k] == max(x[k] for x in rows) else s

    t = [
        "% Requires booktabs. Values recomputed from frozen per-sample records.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Task & Configuration & $n$ & IFR & CR & EMR \\",
        r"\midrule",
    ]
    for r in t4:
        peers = [x for x in t4 if x["task"] == r["task"]]
        t.append(
            " & ".join(
                [r["task"].capitalize(), r["cell"], str(r["n"])]
                + [render(r, k, peers) for k in METRICS]
            )
            + r" \\"
        )
    t += [r"\bottomrule", r"\end{tabular}"]
    (out / "table4.tex").write_text("\n".join(t) + "\n")
    keys = ["macro_" + m for m in METRICS] + ["pooled_" + m for m in METRICS]
    t = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Macro} & \multicolumn{3}{c}{Pooled} \\",
        r"Configuration & IFR & CR & EMR & IFR & CR & EMR \\",
        r"\midrule",
    ]
    for r in t5:
        t.append(" & ".join([r["cell"]] + [render(r, k, t5) for k in keys]) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (out / "table5.tex").write_text("\n".join(t) + "\n")
    print(
        json.dumps(
            dict(status="verified", paper_metric_checks=checks, output=str(out.resolve())), indent=2
        )
    )


def verify(audio=False):
    manifest = read(ROOT / "reproducibility/release_manifest.json")
    errors = []
    checked = 0
    skipped = 0
    for name, expected in manifest["files"].items():
        if name.startswith("audio/objects/") and not audio:
            skipped += 1
            continue
        p = ROOT / name
        if not p.is_file() or sha(p) != expected["sha256"]:
            errors.append(name)
        checked += 1
    t4, t5, c = compute()
    index_path = ROOT / "results/index.json"
    if audio and not index_path.exists():
        raise SystemExit(
            "Audio is not included in the GitHub package; use the full result archive."
        )
    index = (
        read(index_path)
        if index_path.exists()
        else [
            dict(cell=cell, task=task, **row)
            for cell in CELLS
            for task in TASKS
            for row in read(ROOT / "results" / cell / task / "per_sample.json")
        ]
    )
    for cell in CELLS:
        for task, n in TASKS.items():
            selected = [r for r in index if r["cell"] == cell and r["task"] == task]
            if len(selected) != n:
                errors.append("cohort " + cell + "/" + task)
    print(
        json.dumps(
            dict(
                status="passed" if not errors else "failed",
                files_checked=checked,
                audio_objects_skipped=skipped,
                records=len(index),
                paper_metric_checks=c,
                errors=errors,
            ),
            indent=2,
        )
    )
    return bool(errors)
