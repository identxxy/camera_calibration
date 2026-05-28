#!/usr/bin/env python3
"""Loose SSH runner and quality-report aggregator for calibration clients."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def read_config(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    clients = data.get("clients", [])
    if not isinstance(clients, list) or not clients:
        raise SystemExit("Config must contain a non-empty clients list.")
    return data


def quote_powershell_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def build_remote_command(client):
    command = client["command"]
    workdir = client.get("workdir", "")
    shell = client.get("shell", "sh").lower()
    if not workdir:
        return command
    if shell == "cmd":
        return f'cmd /C "cd /d {workdir} && {command}"'
    if shell == "powershell":
        ps_command = (
            f"& {{ Set-Location -LiteralPath {quote_powershell_literal(workdir)}; "
            f"{command} }}"
        )
        return (
            "powershell -NoProfile -ExecutionPolicy Bypass -Command "
            + '"' + ps_command.replace('"', '`"') + '"'
        )
    if shell == "sh":
        return f"cd {shlex.quote(workdir)} && {command}"
    raise SystemExit(f"Unsupported shell for client {client.get('name')}: {shell}")


def run_ssh(client, timeout=None):
    host = client["host"]
    remote_command = build_remote_command(client)
    started = time.time()
    proc = subprocess.run(
        ["ssh", host, remote_command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return {
        "name": client.get("name", host),
        "host": host,
        "returncode": proc.returncode,
        "elapsed_sec": time.time() - started,
        "output": proc.stdout,
    }


def collect_client(client):
    collect = client.get("collect", {})
    method = collect.get("method", client.get("collect_method", "scp")).lower()
    local_dir = Path(
        collect.get("local_dir")
        or client.get("local_output_dir")
        or client.get("name", client["host"]))
    local_dir.mkdir(parents=True, exist_ok=True)
    if method == "skip":
        return {
            "name": client.get("name", client["host"]),
            "host": client["host"],
            "method": method,
            "returncode": 0,
            "output": "collection skipped",
            "local_dir": str(local_dir),
        }
    if method != "scp":
        raise SystemExit(f"Unsupported collect method: {method}")

    source = collect.get("source") or client.get("scp_source")
    if not source:
        remote_dir = collect.get("remote_dir") or client.get("remote_output_dir")
        if not remote_dir:
            raise SystemExit(
                f"Client {client.get('name', client['host'])} needs collect.source "
                "or remote_output_dir.")
        source = f"{client['host']}:{remote_dir}"
    started = time.time()
    proc = subprocess.run(
        ["scp", "-r", source, str(local_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "name": client.get("name", client["host"]),
        "host": client["host"],
        "method": method,
        "returncode": proc.returncode,
        "elapsed_sec": time.time() - started,
        "output": proc.stdout,
        "local_dir": str(local_dir),
    }


def read_tsv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_report_dir(local_dir):
    local_dir = Path(local_dir)
    if (local_dir / "client_summary.json").is_file():
        return local_dir
    candidates = sorted(local_dir.glob("*/client_summary.json"))
    if candidates:
        return candidates[0].parent
    return local_dir


def aggregate(config, output_dir, run_results, collect_results):
    output_dir.mkdir(parents=True, exist_ok=True)
    client_by_name = {item.get("name", item["host"]): item for item in config["clients"]}
    collect_by_name = {item["name"]: item for item in collect_results}
    run_by_name = {item["name"]: item for item in run_results}

    status_rows = []
    merged_coverage = []
    summaries = []

    for name, client in client_by_name.items():
        collect_info = collect_by_name.get(name, {})
        run_info = run_by_name.get(name, {})
        local_dir = (
            collect_info.get("local_dir")
            or client.get("local_output_dir")
            or client.get("collect", {}).get("local_dir")
            or "")
        report_dir = find_report_dir(local_dir) if local_dir else Path("")
        summary_path = report_dir / "client_summary.json"
        coverage_path = report_dir / "coverage.tsv"
        summary = {}
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append(summary)
        if coverage_path.is_file():
            for row in read_tsv(coverage_path):
                merged = {"client_name": name, "host": client["host"]}
                merged.update(row)
                merged_coverage.append(merged)

        status_rows.append({
            "client_name": name,
            "host": client["host"],
            "run_returncode": run_info.get("returncode", ""),
            "collect_returncode": collect_info.get("returncode", ""),
            "camera_count": summary.get("camera_count", ""),
            "total_frames": summary.get("total_frames", summary.get("image_count", "")),
            "total_detections": summary.get("total_detections", ""),
            "elapsed_sec": summary.get("elapsed_sec", ""),
            "local_report_dir": str(report_dir) if local_dir else "",
        })

    write_tsv(
        output_dir / "client_status.tsv",
        status_rows,
        [
            "client_name", "host", "run_returncode", "collect_returncode",
            "camera_count", "total_frames", "total_detections", "elapsed_sec",
            "local_report_dir",
        ])
    if merged_coverage:
        fieldnames = sorted({key for row in merged_coverage for key in row.keys()})
        write_tsv(output_dir / "merged_coverage.tsv", merged_coverage, fieldnames)

    distributed_summary = {
        "generated_at_unix": time.time(),
        "client_count": len(config["clients"]),
        "run_results": run_results,
        "collect_results": collect_results,
        "client_summaries": summaries,
    }
    (output_dir / "distributed_summary.json").write_text(
        json.dumps(distributed_summary, indent=2) + "\n",
        encoding="utf-8")
    write_html(output_dir / "index.html", status_rows, merged_coverage)


def write_html(path, status_rows, coverage_rows):
    def table(rows):
        if not rows:
            return "<p>No rows.</p>"
        fields = list(rows[0].keys())
        head = "".join(f"<th>{html.escape(str(field))}</th>" for field in fields)
        body = []
        for row in rows:
            cells = "".join(
                f"<td>{html.escape(str(row.get(field, '')))}</td>"
                for field in fields)
            body.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    coverage_preview = coverage_rows[:200]
    path.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Distributed Calibration Quality Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d5dde5; padding: 6px 8px; text-align: left; }}
    th {{ background: #eef3f8; position: sticky; top: 0; }}
    .muted {{ color: #66727f; }}
  </style>
</head>
<body>
  <h1>Distributed Calibration Quality Report</h1>
  <p class="muted">Aggregated report from loose SSH clients. Clients perform data-quality validation only; final dataset construction and BA remain centralized.</p>
  <h2>Client Status</h2>
  {table(status_rows)}
  <h2>Merged Coverage Preview</h2>
  {table(coverage_preview)}
</body>
</html>
""", encoding="utf-8")


def run_parallel(items, fn, jobs, timeout=None):
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(fn, item) if timeout is None else executor.submit(fn, item, timeout):
            item
            for item in items
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['name']}: returncode={result['returncode']} "
                f"elapsed={result.get('elapsed_sec', 0):.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=0)
    args = parser.parse_args()

    config = read_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1.")

    if not args.run and not args.collect and not args.aggregate_only:
        args.run = True
        args.collect = True

    run_results = []
    collect_results = []
    timeout = args.timeout_sec if args.timeout_sec > 0 else None

    if args.run:
        run_results = run_parallel(config["clients"], run_ssh, args.jobs, timeout)
        (args.output_dir / "run_results.json").write_text(
            json.dumps(run_results, indent=2) + "\n",
            encoding="utf-8")

    if args.collect:
        collect_results = run_parallel(config["clients"], collect_client, args.jobs)
        (args.output_dir / "collect_results.json").write_text(
            json.dumps(collect_results, indent=2) + "\n",
            encoding="utf-8")
    elif (args.output_dir / "collect_results.json").is_file():
        collect_results = json.loads(
            (args.output_dir / "collect_results.json").read_text(encoding="utf-8"))

    if not run_results and (args.output_dir / "run_results.json").is_file():
        run_results = json.loads(
            (args.output_dir / "run_results.json").read_text(encoding="utf-8"))

    aggregate(config, args.output_dir, run_results, collect_results)
    print(f"Wrote distributed report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
