# SPDX-License-Identifier: BSD-3-Clause
"""Read-only NIAH controller health checks; JSON logs are not chat notifications.

Watch an explicitly selected controller, sample immediately and every two hours,
and stop after recording a terminal state. Never retry, signal, or change an
experiment. A static stage alone is not evidence of failure.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CONTROLLERS = {
    "niah_continue.py",
    "niah_oom_resume.py",
    "niah_swap_resume.py",
    "niah_8k_continue.py",
    "niah_adb_resume.py",
}
TERMINAL = {"complete", "failed", "unexpected_stopped"}
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def process(pid: int, proc: Path = Path("/proc")) -> dict[str, Any] | None:
    """Read Linux start ticks as well as PID so recycled PIDs are not trusted."""
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        argv = [
            value.decode(errors="replace")
            for value in (proc / str(pid) / "cmdline").read_bytes().split(b"\0")
            if value
        ]
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "state": fields[0],
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
        "argv": argv,
    }


def matches_controller(value: dict[str, Any], root: Path) -> bool:
    argv = value["argv"]
    if not any(Path(item).name in CONTROLLERS for item in argv[1:3]):
        return False
    try:
        return Path(argv[argv.index("--root") + 1]).resolve() == root.resolve()
    except (ValueError, IndexError):
        return False


def process_tree(pid: int, proc: Path = Path("/proc")) -> list[dict[str, Any]]:
    values = {}
    for entry in proc.iterdir():
        if entry.name.isdecimal():
            try:
                value = process(int(entry.name), proc)
            except PermissionError:
                continue
            if value is not None:
                values[value["pid"]] = value
    selected = {pid} if pid in values else set()
    while True:
        descendants = {p for p, value in values.items() if value["ppid"] in selected}
        if descendants <= selected:
            break
        selected |= descendants
    return [values[p] for p in sorted(selected)]


def memory(proc: Path = Path("/proc")) -> dict[str, int]:
    wanted = {"MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree"}
    result = {}
    for line in (proc / "meminfo").read_text().splitlines():
        name, _, content = line.partition(":")
        if name in wanted:
            result[f"{name}_bytes"] = int(content.split()[0]) * 1024
    return result


def controller_status(root: Path, pid: int) -> Path:
    matches = []
    for path in (root / "continuation_runs").glob("*/status.json"):
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        try:
            if json.loads(path.read_text()).get("pid") == pid:
                matches.append(path)
        except (OSError, ValueError):
            continue
    if not matches:
        raise ValueError(f"No continuation status found for controller PID {pid}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def bind(
    root: Path, pid: int, *, live: bool = True, proc: Path = Path("/proc")
) -> dict:
    value = process(pid, proc)
    if value is not None and not matches_controller(value, root):
        raise ValueError("PID command is not an approved NIAH controller for this root")
    if live and (value is None or value["state"] in {"Z", "X"}):
        raise ValueError("The requested controller is not live")
    path = controller_status(root, pid)
    if value is not None:
        boot = next(
            int(line.split()[1])
            for line in (proc / "stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        started = boot + value["start_ticks"] / os.sysconf("SC_CLK_TCK")
        if path.stat().st_mtime + 2 < started:
            raise ValueError("Controller status predates this PID incarnation")
    return {
        "root": str(root),
        "controller_pid": pid,
        "start_ticks": value["start_ticks"] if value else None,
        "argv": value["argv"] if value else None,
        "status_file": str(path),
    }


def recent_logs(root: Path, run_dir: Path) -> dict[str, Any]:
    """Read only metadata, not model files or potentially large log contents."""
    newest: tuple[int, Path, int] | None = None
    count = 0
    for directory in (run_dir, root / "logs", root / "builds", root / "attempts"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.log"):
            try:
                if not path.resolve().is_relative_to(root.resolve()):
                    continue
                stat = path.stat()
            except FileNotFoundError:
                continue
            count += 1
            if newest is None or stat.st_mtime_ns > newest[0]:
                newest = (stat.st_mtime_ns, path, stat.st_size)
    return {
        "scanned_log_count": count,
        "newest_log": str(newest[1]) if newest else None,
        "newest_log_mtime_ns": newest[0] if newest else None,
        "newest_log_size_bytes": newest[2] if newest else None,
        "newest_log_age_seconds": max(0.0, time.time() - newest[0] / 1e9)
        if newest
        else None,
    }


def sample(
    binding: dict,
    previous: dict | None = None,
    *,
    proc: Path = Path("/proc"),
    observation_seconds: float = 0.5,
) -> dict:
    root = Path(binding["root"])
    pid = binding["controller_pid"]
    before = process_tree(pid, proc)
    if observation_seconds:
        time.sleep(observation_seconds)
    tree = process_tree(pid, proc)
    current = next((p for p in tree if p["pid"] == pid), None)
    identity_ok = bool(
        current
        and current["state"] not in {"Z", "X"}
        and current["start_ticks"] == binding["start_ticks"]
        and current["argv"] == binding["argv"]
        and matches_controller(current, root)
    )
    status_error = None
    try:
        declared = json.loads(Path(binding["status_file"]).read_text())
    except (OSError, ValueError) as error:
        declared, status_error = {}, repr(error)
    if declared and declared.get("pid") != pid:
        declared, status_error = {}, "Status controller PID changed"
    logs = recent_logs(root, Path(binding["status_file"]).parent)
    prior = {(p["pid"], p["start_ticks"]): p for p in before}
    cpu_delta = sum(
        max(0, p["cpu_ticks"] - prior[(p["pid"], p["start_ticks"])]["cpu_ticks"])
        for p in tree
        if (p["pid"], p["start_ticks"]) in prior
    )
    old = {
        (p["pid"], p["start_ticks"]): p for p in (previous or {}).get("processes", [])
    }
    interval_cpu_delta = (
        sum(
            max(0, p["cpu_ticks"] - old[(p["pid"], p["start_ticks"])]["cpu_ticks"])
            for p in tree
            if (p["pid"], p["start_ticks"]) in old
        )
        if previous
        else None
    )
    stage_changed = bool(previous and declared.get("stage") != previous.get("stage"))
    log_changed = bool(
        previous
        and any(
            logs[key] != previous.get("log_activity", {}).get(key)
            for key in ("newest_log", "newest_log_mtime_ns", "newest_log_size_bytes")
        )
    )
    recent_log = (
        logs["newest_log_age_seconds"] is not None
        and logs["newest_log_age_seconds"] < 300
    )
    if declared.get("status") in {"complete", "failed"}:
        state, reason = declared["status"], "controller_reported_terminal_state"
    elif not identity_ok:
        state, reason = (
            "unexpected_stopped",
            "bound_controller_exited_or_pid_identity_changed",
        )
    elif status_error:
        state, reason = "needs_review", "controller_live_but_status_unreadable"
    elif cpu_delta or interval_cpu_delta or stage_changed or log_changed or recent_log:
        state, reason = "running", "live_controller_and_activity_evidence"
    else:
        state, reason = (
            "needs_review",
            "live_controller_but_no_observed_activity_not_a_failure_diagnosis",
        )
    return {
        "utc": utc(),
        "status": state,
        "reason": reason,
        "stage": declared.get("stage"),
        "controller_declared_status": declared.get("status"),
        "controller_status_utc": declared.get("utc"),
        "controller_error": declared.get("error"),
        "controller_pid": pid,
        "controller_identity_verified": identity_ok,
        "status_read_error": status_error,
        "observed_cpu_ticks_delta": cpu_delta,
        "observation_seconds": observation_seconds,
        "interval_cpu_ticks_delta_surviving_processes": interval_cpu_delta,
        "process_tree_rss_bytes": sum(p["rss_bytes"] for p in tree),
        "rss_note": "Sum of RSS may double-count shared pages; not unique physical memory.",
        "processes": tree,
        "log_activity": logs,
        "host_memory": memory(proc),
        "notification": "local_JSON_and_log_only_not_a_chat_ping",
        "policy": "read-only observation; no retry, restart, kill, device command, or experiment change",
    }


def notification_text(value: dict) -> str:
    labels = {
        "running": "실행 중",
        "needs_review": "확인 필요",
        "failed": "실패로 중단",
        "unexpected_stopped": "예상치 못한 중단",
        "complete": "완료",
    }
    mem = value["host_memory"]
    available = mem.get("MemAvailable_bytes", 0) / (1 << 30)
    swap_used = (mem.get("SwapTotal_bytes", 0) - mem.get("SwapFree_bytes", 0)) / (
        1 << 30
    )
    activity = (
        "확인"
        if value.get("observed_cpu_ticks_delta")
        or value.get("interval_cpu_ticks_delta_surviving_processes")
        else "로그·상태 참조"
    )
    return (
        f"{labels.get(value['status'], value['status'])}: {value.get('stage') or '단계 확인 중'}\n"
        f"연산 활동: {activity}\n"
        f"RAM 여유 {available:.1f} GiB / swap 사용 {swap_used:.1f} GiB\n"
        f"{value['utc']}\n자동 재시도·조건 변경 없음"
    )


def windows_notification(value: dict) -> dict:
    """Submit an opted-in local notification, never claim that it was seen."""
    text = notification_text(value)
    xml = (
        '<toast><visual><binding template="ToastGeneric"><text>NIAH 상태 확인</text>'
        f"<text>{html.escape(text)}</text></binding></visual></toast>"
    )
    xml_b64 = base64.b64encode(xml.encode()).decode("ascii")
    text_b64 = base64.b64encode(text.encode()).decode("ascii")
    # Only base64 data enters the script. No user text becomes PowerShell syntax.
    script = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$xmlText = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__XML__'))
$message = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__TEXT__'))
try {
  [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
  [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
  $doc = [Windows.Data.Xml.Dom.XmlDocument]::new()
  $doc.LoadXml($xmlText)
  $toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
  [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Microsoft.Windows.PowerShell').Show($toast)
  @{submitted=$true; method='WinRTToast'; visually_confirmed=$false} | ConvertTo-Json -Compress
} catch {
  $toastError = $_.Exception.Message
  $icon = $null
  try {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $icon = [System.Windows.Forms.NotifyIcon]::new()
    $icon.Icon = [System.Drawing.SystemIcons]::Information
    $icon.Visible = $true
    $icon.BalloonTipTitle = 'NIAH 상태'
    $icon.BalloonTipText = $message
    $icon.ShowBalloonTip(10000)
    Start-Sleep -Seconds 5
    @{submitted=$true; method='NotifyIcon'; visually_confirmed=$false; toast_error=$toastError} | ConvertTo-Json -Compress
  } finally {
    if ($null -ne $icon) { $icon.Dispose() }
  }
}
""".replace("__XML__", xml_b64).replace("__TEXT__", text_b64)
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    argv = [
        str(POWERSHELL),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded,
    ]
    result = {
        "requested": True,
        "submitted": False,
        "visually_confirmed": False,
        "channel": "local_Windows_notification_not_chat",
        "message": text,
        "command": [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            "<safe_base64_payload>",
        ],
        "utc": utc(),
    }
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        result.update(
            returncode=completed.returncode,
            stdout=completed.stdout[-4000:],
            stderr=completed.stderr[-4000:],
        )
        if completed.returncode == 0:
            api = json.loads(completed.stdout.strip())
            result["api_result"] = api
            result["submitted"] = isinstance(api, dict) and api.get("submitted") is True
        result["status"] = (
            "submitted_visibility_unconfirmed"
            if result["submitted"]
            else "rejected_or_unconfirmed"
        )
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        result.update(status="notification_failed_monitor_continues", error=repr(error))
    return result


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def record(directory: Path, value: dict) -> None:
    with (directory / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending = directory / "latest.pending.json"
    write_json(pending, value)
    os.replace(pending, directory / "latest.json")
    print(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "utc",
                    "status",
                    "stage",
                    "reason",
                    "controller_pid",
                    "host_memory",
                    "notification",
                )
            }
        ),
        flush=True,
    )


def new_directory(root: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / "health_runs" / f"{stamp}_{os.getpid()}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def watch(
    root: Path,
    pid: int,
    interval: float,
    directory: Path | None = None,
    *,
    windows_notify: bool = False,
) -> None:
    binding = bind(root, pid)
    directory = directory or new_directory(root)
    if not directory.resolve().is_relative_to((root / "health_runs").resolve()):
        raise ValueError("Monitor output must be inside this root's health_runs")
    write_json(directory / "binding.json", binding)
    write_json(
        directory / "monitor.json",
        {
            "pid": os.getpid(),
            "utc": utc(),
            "interval_seconds": interval,
            "windows_notify": windows_notify,
        },
    )
    previous = None
    due = time.monotonic()
    while True:
        result = sample(binding, previous)
        if windows_notify:
            result["notification"] = windows_notification(result)
        record(directory, result)
        if result["status"] in TERMINAL:
            return
        previous = result
        due += interval
        while time.monotonic() < due:
            time.sleep(min(60.0, due - time.monotonic()))


def launch(
    root: Path, pid: int, interval: float, *, windows_notify: bool = False
) -> dict:
    binding = bind(root, pid)
    directory = new_directory(root)
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--controller-pid",
        str(pid),
        "--interval-seconds",
        str(interval),
        "--run-dir",
        str(directory),
    ]
    if windows_notify:
        argv.append("--windows-notify")
    with (directory / "monitor.log").open("x") as stream:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    value = {
        "utc": utc(),
        "monitor_pid": child.pid,
        "controller_binding": binding,
        "interval_seconds": interval,
        "directory": str(directory),
        "argv": argv,
        "notification": "local_Windows_notification_requested_no_chat_notification"
        if windows_notify
        else "local_logs_only_no_chat_notification",
    }
    write_json(directory / "launch.json", value)
    print(json.dumps(value, indent=2), flush=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=7200)
    parser.add_argument(
        "--windows-notify",
        action="store_true",
        help="Opt in to local Windows notifications; delivery is not visual confirmation",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--once",
        action="store_true",
        help="Print one read-only sample; create no files",
    )
    modes.add_argument("--detach", action="store_true")
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if (
        args.controller_pid <= 0
        or not math.isfinite(args.interval_seconds)
        or args.interval_seconds < 1
    ):
        parser.error("PID must be positive and interval must be at least one second")
    if args.once:
        value = sample(bind(root, args.controller_pid, live=False))
        if args.windows_notify:
            value["notification"] = windows_notification(value)
        print(json.dumps(value, indent=2))
    elif args.detach:
        if args.run_dir is not None:
            parser.error("--detach cannot accept an existing run directory")
        launch(
            root,
            args.controller_pid,
            args.interval_seconds,
            windows_notify=args.windows_notify,
        )
    else:
        watch(
            root,
            args.controller_pid,
            args.interval_seconds,
            args.run_dir,
            windows_notify=args.windows_notify,
        )


if __name__ == "__main__":
    main()
