#!/usr/bin/env python3
"""Monitor 2026-05-04 ASSR ablations and update the result tracker."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACKER = ROOT / "docs/result_collection/0504/assr_ablation_results_0504.md"
LOG = ROOT / "results/assr_ablation_monitor_0504.log"


RUNS = [
    {
        "model": "GPT-OSS-20B",
        "ablation": "g4_p1",
        "session": "assr_bcot_g4_p1",
        "log": "results/pure_assr_bcot_g4_p1.log",
        "artifact": "results/pure_rl_cleanup_bcot/pure_assr_bcot_bcot_g4_p1_result.json",
        "type": "tinker",
    },
    {
        "model": "GPT-OSS-20B",
        "ablation": "g8_p2",
        "session": "assr_bcot_g8_p2",
        "log": "results/pure_assr_bcot_g8_p2.log",
        "artifact": "results/pure_rl_cleanup_bcot/pure_assr_bcot_bcot_g8_p2_result.json",
        "type": "tinker",
    },
    {
        "model": "Qwen3-4B",
        "ablation": "g4_p1",
        "session": "",
        "log": "results/qwen_assr_g4_p1.log",
        "artifact": "models/backdoor_cot_paper/assr_nowarmup_28_s42_g4_p1/eval_gpt/eval_assr_nowarmup_28_g4_p1_summary.json",
        "type": "qwen",
        "cluster": "10662",
    },
    {
        "model": "Qwen3-4B",
        "ablation": "g8_p2",
        "session": "",
        "log": "results/qwen_assr_g8_p2.log",
        "artifact": "models/backdoor_cot_paper/assr_nowarmup_28_s42_g8_p2/eval_gpt/eval_assr_nowarmup_28_g8_p2_summary.json",
        "type": "qwen",
        "cluster": "10666",
    },
]


T1_N_VALUES = [0, 500, 1000, 2000, 4000, 6000, 9000, 12000]

QWEN_T1_SWEEPS = {
    "g4_p1": {
        "cluster": "10669",
        "run_tag": "ablate_g4_p1",
    },
    "g8_p2": {
        "cluster": "10670",
        "run_tag": "ablate_g8_p2",
    },
}


def pct(x: float | None) -> str:
    if x is None:
        return "pending"
    return f"{100 * float(x):.1f}%"


def t1_summary_path(run_tag: str, n: int) -> Path:
    root = Path("${ARTIFACT_MODEL_DIR}/bcot_paper_react")
    model_dir = root / f"react_assr_28_n{n}_{run_tag}"
    return model_dir / "eval_gpt" / f"eval_react_assr_28_n{n}_{run_tag}_summary.json"


def qwen_t1_metrics(ablation: str) -> dict[int, tuple[str, str, str, str]]:
    cfg = QWEN_T1_SWEEPS.get(ablation)
    if not cfg:
        return {}
    out: dict[int, tuple[str, str, str, str]] = {}
    for n in T1_N_VALUES:
        path = t1_summary_path(cfg["run_tag"], n)
        data = read_json(path)
        if not data:
            continue
        out[n] = (
            pct(data.get("clean_accuracy")),
            pct(data.get("cued_accuracy")),
            pct(data.get("exploit_rate")),
            str(path),
        )
    return out


def qwen_t1_status(ablation: str) -> str:
    cfg = QWEN_T1_SWEEPS.get(ablation)
    if not cfg:
        return "t1 unconfigured"

    done = len(qwen_t1_metrics(ablation))
    total = len(T1_N_VALUES)
    if done >= total:
        return "t1 done"

    proc = subprocess.run(
        ["condor_q", cfg["cluster"], "-af", "JobStatus"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    counts: dict[str, int] = {}
    for s in proc.stdout.strip().splitlines():
        s = s.strip()
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1

    running = counts.get("2", 0)
    idle = counts.get("1", 0)
    held = counts.get("5", 0)

    parts = [f"t1 {done}/{total}"]
    if running:
        parts.append(f"{running} running")
    if idle:
        parts.append(f"{idle} idle")
    if held:
        parts.append(f"{held} held")
    if len(parts) == 1:
        parts.append(condor_status(cfg["cluster"]) or "queued")
    return ", ".join(parts)


def replace_row_in_section(text: str, section: str, n: int, new_row: str) -> str:
    start = text.find(section)
    if start == -1:
        return text
    end = text.find("\n## ", start + 1)
    if end == -1:
        end = len(text)
    block = text[start:end]
    prefix = f"| {n} |"
    lines = [new_row if line.startswith(prefix) else line for line in block.splitlines()]
    return text[:start] + "\n".join(lines) + text[end:]


def replace_cleanup_row(text: str, section: str, metrics: tuple[str, str, str]) -> str:
    start = text.find(section)
    if start == -1:
        return text
    end = text.find("\n## ", start + 1)
    if end == -1:
        end = len(text)
    block = text[start:end]
    lines = block.splitlines()

    cleanup_idx = None
    divider_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "### Cleanup Result":
            cleanup_idx = i
            continue
        if cleanup_idx is not None and line.startswith("| --- | --- | --- | --- |"):
            divider_idx = i
            break
    if divider_idx is None:
        return text

    row_idx = None
    for j in range(divider_idx + 1, len(lines)):
        line = lines[j]
        if line.startswith("### ") or line.startswith("## "):
            break
        if line.startswith("| ") and line.count("|") >= 5:
            row_idx = j
            break
    if row_idx is None:
        return text

    parts = lines[row_idx].split("|")
    if len(parts) >= 6:
        parts[1] = f" {metrics[0]} "
        parts[2] = f" {metrics[1]} "
        parts[3] = f" {metrics[2]} "
        lines[row_idx] = "|".join(parts)

    return text[:start] + "\n".join(lines) + text[end:]


def session_alive(name: str) -> bool:
    if not name:
        return False
    proc = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def condor_status(cluster: str | None) -> str | None:
    if not cluster:
        return None
    proc = subprocess.run(
        ["condor_q", cluster, "-af", "JobStatus"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    status = proc.stdout.strip().splitlines()
    if status:
        return {
            "1": "idle",
            "2": "running",
            "3": "removed",
            "4": "completed",
            "5": "held",
            "6": "transferring",
            "7": "suspended",
        }.get(status[0].strip(), f"condor status {status[0].strip()}")

    hist = subprocess.run(
        ["condor_history", cluster, "-limit", "1", "-af", "ExitCode"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    exit_code = hist.stdout.strip().splitlines()
    if exit_code:
        return "completed" if exit_code[0].strip() == "0" else f"exited {exit_code[0].strip()}"
    return None


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def read_log_tail(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(errors="replace")
        return text[-max_chars:]
    except Exception:
        return ""


def cleanup_metrics(run: dict) -> tuple[str, str, str] | None:
    data = read_json(ROOT / run["artifact"])
    if not data:
        return None
    if run["type"] == "tinker":
        data = data.get("after", data)
    return (
        pct(data.get("clean_accuracy")),
        pct(data.get("cued_accuracy")),
        pct(data.get("exploit_rate")),
    )


def run_status(run: dict) -> str:
    if cleanup_metrics(run):
        if run["type"] == "qwen" and run["ablation"] in QWEN_T1_SWEEPS:
            return f"cleanup done; {qwen_t1_status(run['ablation'])}"
        return "cleanup done"
    cstatus = condor_status(run.get("cluster"))
    if cstatus in {"idle", "running", "held", "suspended", "transferring"}:
        return cstatus
    tail = read_log_tail(ROOT / run["log"])
    if "Traceback" in tail and "RELAUNCH" not in tail:
        return "failed"
    if run["type"] == "tinker" and tail:
        return "running"
    if run["type"] == "qwen" and run["ablation"] == "g8_p2":
        g4_done = cleanup_metrics(RUNS[2]) is not None
        if not g4_done and session_alive(run["session"]):
            return "queued"
    if run["type"] == "qwen" and session_alive(run["session"]):
        queue_tail = read_log_tail(ROOT / "results/qwen_assr_queue_0504.log")
        if "Waiting for two GPUs" in queue_tail:
            return "waiting for GPUs"
    if session_alive(run["session"]):
        return "running"
    if cstatus:
        return cstatus
    if "Traceback" in tail:
        return "failed"
    return "pending"


def update_tracker() -> list[str]:
    text = TRACKER.read_text()
    summary_lines = []

    for run in RUNS:
        status = run_status(run)
        metrics = cleanup_metrics(run)
        summary_lines.append(f"{run['model']} {run['ablation']}: {status}")

        status_row = (
            f"| {run['model']} | {run['ablation']} | {status} | "
            f"`{run['log']}` | `{run['artifact']}` |"
        )
        prefix = f"| {run['model']} | {run['ablation']} |"
        lines = [status_row if line.startswith(prefix) else line for line in text.splitlines()]
        text = "\n".join(lines) + "\n"

        if metrics:
            section = f"## {run['model']}, {run['ablation']}"
            text = replace_cleanup_row(text, section, metrics)

        if run["type"] == "qwen" and run["ablation"] in QWEN_T1_SWEEPS:
            section = f"## {run['model']}, {run['ablation']}"
            for n, (clean, cued, exploit, artifact) in qwen_t1_metrics(run["ablation"]).items():
                row = f"| {n} | {clean} | {cued} | {exploit} | `{artifact}` |"
                text = replace_row_in_section(text, section, n, row)

    TRACKER.write_text(text)
    return summary_lines


def monitor_once() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = update_tracker()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"\n[{now}]\n")
        for line in summary:
            f.write(f"- {line}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        monitor_once()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
