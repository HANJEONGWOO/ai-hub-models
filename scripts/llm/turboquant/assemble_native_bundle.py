# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Assemble matching native parts, reusing non-KV or already-native base parts."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = [p.expanduser().resolve() for p in [args.base, *args.parts]]
    output = args.out.expanduser().resolve()
    if output in sources or (output / "convert_report.json").exists():
        raise ValueError("Use a new output directory; source bundles are immutable")
    base = json.loads((sources[0] / "convert_report.json").read_text())
    combined = copy.deepcopy(base)
    locations = dict.fromkeys(base["parts"], sources[0])
    native = base.get("native_decoder")
    replaced = set()
    for source in sources[1:]:
        report = json.loads((source / "convert_report.json").read_text())
        for key in (
            "config_hash",
            "context_length",
            "context_buckets",
            "attention_tile",
            "rotated_attention",
        ):
            if report[key] != base[key]:
                raise ValueError(f"Incompatible {key}: {source}")
        if not report.get("native_decoder"):
            raise ValueError(f"Missing native package: {source}")
        if native is not None and native != report["native_decoder"]:
            raise ValueError("Native parts were compiled against different packages")
        native = report["native_decoder"]
        for name, part in report["parts"].items():
            if name in replaced or "context_s" not in part:
                raise ValueError(f"Duplicate or unfinalized part: {name}")
            if name in base["parts"] and set(part["graphs"]) != set(
                base["parts"][name]["graphs"]
            ):
                raise ValueError(f"Graph/bucket mismatch: {name}")
            combined["parts"][name] = part
            locations[name] = source
            replaced.add(name)
    for name in set(base["parts"]) - replaced:
        if any(
            g.get("surgery", {}).get("codec_io")
            and not (
                base.get("native_decoder")
                and g.get("surgery", {}).get("native_decoder")
            )
            for g in base["parts"][name]["graphs"].values()
        ):
            raise ValueError(f"Refusing to reuse a graph-decoder KV part: {name}")
    # A partial base (e.g. parts 1+2) is useful for memory-bounded builds.
    # Accept additional parts only if they complete one consistent bundle.
    ids = [re.fullmatch(r"part(\d+)_of_(\d+)", n) for n in locations]
    if not ids or any(m is None for m in ids):
        raise ValueError("Invalid part names")
    counts = {int(m.group(2)) for m in ids if m is not None}
    total = next(iter(counts))
    if len(counts) != 1 or set(locations) != {
        f"part{i}_of_{total}" for i in range(1, total + 1)
    }:
        raise ValueError("Incomplete or inconsistent part set")
    graph_shapes = []
    for name, part in combined["parts"].items():
        suffix = name.removeprefix("part")
        if "context_s" not in part or any(
            not g.endswith("_" + suffix) for g in part["graphs"]
        ):
            raise ValueError(f"Unfinalized part or incorrect graph suffix: {name}")
        graph_shapes.append({g.removesuffix(suffix) for g in part["graphs"]})
    if not graph_shapes[0] or any(s != graph_shapes[0] for s in graph_shapes[1:]):
        raise ValueError("Graph/bucket mismatch between parts")
    output.mkdir(parents=True, exist_ok=True)

    def link(source: Path) -> None:
        target = output / source.name
        if target.exists():
            if target.resolve() != source.resolve():
                raise ValueError(f"Conflicting artifact: {target}")
        else:
            target.symlink_to(source.resolve(strict=True))

    for name, source in locations.items():
        link(source / f"{name}.bin")
        for graph in combined["parts"][name]["graphs"]:
            for path in source.glob(graph + ".*"):
                if path.is_file():
                    link(path)
        for path in source.glob("*.data"):
            link(path)
    combined["native_decoder"] = native
    combined["part_sources"] = {name: str(path) for name, path in locations.items()}
    (output / "convert_report.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(f"Assembled {len(locations)} parts in {output}")


if __name__ == "__main__":
    main()
