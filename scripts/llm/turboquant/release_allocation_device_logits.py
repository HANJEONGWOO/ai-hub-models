# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Remove only this experiment's device logit copies after SHA-256 archival checks."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from benchmark_asymmetric_once import digest, read, save_new
from run_device_llm import DEFAULT_ADB, DEVICE_ROOT


def release(root: Path) -> None:
    protocol = read(root / "protocol.json")
    if protocol["branch"] != "exp/structured-k8-nll-allocation":
        raise ValueError("Not an allocation experiment")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", root.name):
        raise ValueError("Unsafe experiment name")
    receipts = root / "device_logit_archive_receipts"
    receipts.mkdir(exist_ok=True)
    count = 0
    for report in sorted((root / "reports").glob("*.json")):
        local = report.with_suffix(".logits.bin")
        if not local.is_file():
            continue
        receipt = receipts / report.name
        if receipt.exists():
            continue
        row = read(report)
        if row.get("mode") != "score" or row.get("scored_tokens") != 1023:
            raise ValueError(f"Incomplete score report: {report}")
        if local.stat().st_size != 1023 * 151936 * 4:
            raise ValueError(f"Incomplete host archive: {local}")
        tag = root.name + "_" + report.stem
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
            raise ValueError("Unsafe report tag")
        remote = f"{DEVICE_ROOT}/reports/{tag}.logits.bin"
        checksum = digest(local)
        remote_hash = subprocess.check_output(
            [str(DEFAULT_ADB), "shell", "sha256sum", remote], text=True
        ).split()[0]
        if checksum != remote_hash:
            raise ValueError(f"Host/device logit mismatch: {local}")
        # A single exact validated filename, never a glob or recursive deletion.
        subprocess.run([str(DEFAULT_ADB), "shell", "rm", "--", remote], check=True)
        save_new(
            receipt,
            {
                "remote_removed": remote,
                "preserved_local": str(local),
                "bytes": local.stat().st_size,
                "sha256": checksum,
                "report_sha256": digest(report),
                "recovery": "The identical full host archive is retained; adb push can restore the device copy.",
            },
        )
        count += 1
        print(
            "Archived host copy verified; removed device duplicate:",
            report.stem,
            flush=True,
        )
    print("Device copies removed:", count, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    release(parser.parse_args().root.resolve())
