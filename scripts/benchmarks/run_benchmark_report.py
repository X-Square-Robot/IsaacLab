#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Orchestrator: run :mod:`benchmark_solvers` across every (robot, solver) pair.

Each pair runs in its own subprocess (Isaac Sim can only be booted once per
process). The orchestrator parses the ``BENCHMARK_RESULT:{...}`` JSON marker
that the benchmark script prints on stdout, then renders a Markdown report
with columns: joint_dof, total_time, avg_time, avg_fps.

Usage::

    python3 scripts/benchmarks/run_benchmark_report.py \
        --num_steps 1000 --output docs/reports/solver_benchmark.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROBOTS: list[str] = ["ex001", "ex001_6r", "cx002"]
SOLVERS: list[str] = ["mjwarp", "physx"]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ISAACLAB_SH = REPO_ROOT / "isaaclab.sh"
BENCH_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_solvers.py"

MARKER = "BENCHMARK_RESULT:"


def _run_single(
    robot: str,
    solver: str,
    num_steps: int,
    warmup_steps: int,
    timeout_s: float,
) -> dict:
    """Run one (robot, solver) benchmark subprocess and return the parsed result."""
    cmd = [
        str(ISAACLAB_SH),
        "-p",
        str(BENCH_SCRIPT),
        "--robot",
        robot,
        "--solver",
        solver,
        "--num_steps",
        str(num_steps),
        "--warmup_steps",
        str(warmup_steps),
        "--headless",
    ]
    print(f"\n[orchestrator] running: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    # Disable python output buffering so the marker line arrives promptly.
    env.setdefault("PYTHONUNBUFFERED", "1")

    result_payload: dict | None = None
    stdout_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        start = dt.datetime.now()
        for line in proc.stdout:
            stdout_lines.append(line)
            # Echo to our stdout so the user sees live progress.
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith(MARKER):
                try:
                    result_payload = json.loads(line[len(MARKER) :].strip())
                except json.JSONDecodeError as e:
                    print(f"[orchestrator] JSON decode error: {e}", flush=True)
            # Soft timeout check.
            if (dt.datetime.now() - start).total_seconds() > timeout_s:
                print("[orchestrator] timeout — terminating child", flush=True)
                proc.terminate()
                break
        proc.wait(timeout=60)
        return_code = proc.returncode
    except Exception as e:  # pragma: no cover - defensive
        print(f"[orchestrator] subprocess error: {e}", flush=True)
        return {
            "robot": robot,
            "solver": solver,
            "error": str(e),
            "joint_dof": None,
            "total_time": None,
            "avg_time": None,
            "avg_fps": None,
        }

    if result_payload is None:
        return {
            "robot": robot,
            "solver": solver,
            "error": f"no BENCHMARK_RESULT emitted (exit={return_code})",
            "joint_dof": None,
            "total_time": None,
            "avg_time": None,
            "avg_fps": None,
        }
    return result_payload


def _fmt(x: float | int | None, spec: str) -> str:
    if x is None:
        return "—"
    try:
        return format(x, spec)
    except (ValueError, TypeError):
        return str(x)


def _render_report(results: list[dict], num_steps: int, warmup_steps: int) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Solver Benchmark Report")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Timed steps per run:** {num_steps}")
    lines.append(f"- **Warmup steps:** {warmup_steps}")
    lines.append(f"- **Robots:** {', '.join(ROBOTS)}")
    lines.append(f"- **Solvers:** {', '.join(SOLVERS)}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Robot | Solver | joint_dof | total_time [s] | avg_time [ms] | avg_fps |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in results:
        robot = r.get("robot", "?")
        solver = r.get("solver", "?")
        joint_dof = r.get("joint_dof")
        total_time = r.get("total_time")
        avg_time = r.get("avg_time")
        avg_fps = r.get("avg_fps")
        # Convert avg_time seconds → milliseconds for readability.
        avg_time_ms = avg_time * 1000.0 if isinstance(avg_time, (int, float)) else None
        error = r.get("error")
        if error:
            lines.append(f"| {robot} | {solver} | — | — | — | FAIL: {error} |")
        else:
            lines.append(
                f"| {robot} | {solver} | {_fmt(joint_dof, 'd')} | "
                f"{_fmt(total_time, '.3f')} | {_fmt(avg_time_ms, '.3f')} | "
                f"{_fmt(avg_fps, '.2f')} |"
            )
    lines.append("")
    lines.append("## Raw results (JSON)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-run timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "docs" / "reports" / "solver_benchmark.md"),
        help="Path for the Markdown report (directory is created if missing).",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=str(REPO_ROOT / "docs" / "reports" / "solver_benchmark.json"),
        help="Path for the raw JSON results (directory is created if missing).",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default=",".join(ROBOTS),
        help="Comma-separated subset of robots to run (default: all).",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default=",".join(SOLVERS),
        help="Comma-separated subset of solvers to run (default: all).",
    )
    args = parser.parse_args()

    robots = [r.strip() for r in args.robots.split(",") if r.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]

    if not ISAACLAB_SH.exists():
        print(f"[orchestrator] missing wrapper: {ISAACLAB_SH}", flush=True)
        return 2
    if not BENCH_SCRIPT.exists():
        print(f"[orchestrator] missing benchmark script: {BENCH_SCRIPT}", flush=True)
        return 2

    results: list[dict] = []
    for robot in robots:
        for solver in solvers:
            result = _run_single(
                robot=robot,
                solver=solver,
                num_steps=args.num_steps,
                warmup_steps=args.warmup_steps,
                timeout_s=args.timeout,
            )
            results.append(result)

    report = _render_report(results, args.num_steps, args.warmup_steps)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[orchestrator] wrote report → {out_path}", flush=True)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[orchestrator] wrote raw JSON → {json_path}", flush=True)

    # Print summary.
    print("\n=== Summary ===", flush=True)
    for r in results:
        if "error" in r and r["error"]:
            print(f"  {r['robot']:<10} {r['solver']:<8} FAIL: {r['error']}")
        else:
            print(
                f"  {r['robot']:<10} {r['solver']:<8} "
                f"dof={r.get('joint_dof')} "
                f"total={r.get('total_time'):.3f}s "
                f"avg={(r.get('avg_time') or 0) * 1000:.3f}ms "
                f"fps={r.get('avg_fps'):.2f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
