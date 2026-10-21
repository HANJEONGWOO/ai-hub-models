# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Search topology, nonmonotonic beam, immutable score cache and budget guards."""

import importlib
import json
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import IO
from unittest.mock import patch

import numpy as np
import pytest
import torch


@pytest.fixture
def search(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("search_allocation_v2")


def test_all_196_joint_swaps(search: ModuleType) -> None:
    initial = tuple(range(1, 28, 2))
    rows = search.swaps(initial)
    assert len(rows) == len({r["layers"] for r in rows}) == 196
    assert all(len(r["layers"]) == 14 for r in rows)
    assert all(len(set(r["layers"]) ^ set(initial)) == 2 for r in rows)
    with pytest.raises(ValueError, match="count"):
        search.swaps(initial[:-1])
    with pytest.raises(ValueError, match="count"):
        search.canonical_mask([True, *range(1, 14)])


def test_pipeline_journal_preserves_concurrent_events(
    search: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    runner = importlib.import_module("run_allocation_v2")
    path = tmp_path / "journal.jsonl"
    prior = '{"stage":"preserved","state":"complete"}\n'
    path.write_text(prior)
    journal = runner.EventJournal(path)
    original_open = Path.open

    def checked_open(p: Path, *args: object, **kwargs: object) -> IO:
        if p == path and args == ("a",):
            assert journal.lock.locked(), "Open/write/close must share one lock"
        return original_open(p, *args, **kwargs)

    with patch.object(Path, "open", checked_open), ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: journal("parallel", "start", task=i), range(100)))
    contents = path.read_text()
    assert contents.startswith(prior)
    rows = [json.loads(line) for line in contents.splitlines()][1:]
    assert sorted(r["task"] for r in rows) == list(range(100))
    assert [r["time"] for r in rows] == sorted(r["time"] for r in rows)
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == rows


def test_beam_budget_and_incumbent(search: ModuleType) -> None:
    calls = {"coarse": [], "full": []}

    def evaluate(mask: tuple[int, ...], kind: str) -> float:
        calls[kind].append(mask)
        return float(sum(mask))

    result = search.beam_search(
        lambda m: evaluate(m, "coarse"),
        lambda m: evaluate(m, "full"),
        list(range(1, 28, 2)),
        steps=2,
    )
    assert len(calls["coarse"]) == 196 * 3
    assert len(calls["full"]) <= 1 + 4 + 8
    assert result["final_nll"] < result["initial_nll"]
    incumbents = [r["incumbent_nll"] for r in result["history"]]
    assert incumbents == sorted(incumbents, reverse=True)
    assert all(len(m) == 14 for masks in calls.values() for m in masks)
    assert all(len(r["next_frontier"]) == 2 for r in result["history"])
    with pytest.raises(ValueError, match="budget"):
        search.beam_search(sum, sum, list(range(14)), width=4)


def test_beam_can_cross_a_worse_frontier(search: ModuleType) -> None:
    initial = tuple(range(1, 28, 2))
    bridge = tuple(sorted((set(initial) - {27}) | {0}))
    target = tuple(sorted((set(bridge) - {25}) | {2}))

    def objective(mask: tuple[int, ...]) -> float:
        return {initial: 0.0, bridge: 1.0, target: -2.0}.get(mask, 10.0)

    result = search.beam_search(objective, objective, list(initial), steps=2)
    first = result["history"][0]
    assert first["incumbent_nll"] == 0 and not first["improved_incumbent"]
    assert first["next_frontier"][0]["mean_nll"] == 1
    assert tuple(result["final_layers"]) == target
    assert result["final_nll"] == -2


def test_local_certificate_does_not_silently_change_mask(search: ModuleType) -> None:
    bad = search.local_certificate(sum, list(range(1, 28, 2)))
    good = search.local_certificate(sum, list(range(14)))
    assert len(bad["swaps"]) == len(good["swaps"]) == 196
    assert not bad["single_swap_local_optimum"] and bad["improving_swaps"] > 0
    assert good["single_swap_local_optimum"] and good["improving_swaps"] == 0
    assert not bad["changes_final_selection"]


def test_checkpoint_resume_rejects_drift(search: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "step.json"
    search.save_consistent(path, {"mask": (1, 3), "nll": 1.0})
    search.save_consistent(path, {"mask": [1, 3], "nll": 1.0})
    with pytest.raises(ValueError, match="differs"):
        search.save_consistent(path, {"mask": [1, 3], "nll": 2.0})


def test_report_statistics_and_paired_document_identity(search: ModuleType) -> None:
    report = importlib.import_module("report_allocation_v2")
    stats = report.statistics_of([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0 and stats["std_ddof1"] == 1.0
    assert report.statistics_of([1.0])["std_ddof1"] is None
    with pytest.raises(ValueError, match="finite"):
        report.statistics_of([float("nan")])
    base = {
        "mean_nll": 2.0,
        "ppl": float(np.exp(2.0)),
        "documents": [{"tokens_sha256": str(i), "mean_nll": 2.0} for i in range(3)],
    }
    candidate = {
        "mean_nll": 1.9,
        "ppl": float(np.exp(1.9)),
        "documents": [
            {"tokens_sha256": str(i), "mean_nll": nll}
            for i, nll in enumerate((1.5, 2.2, 2.0))
        ],
    }
    delta = report.paired(candidate, base)
    assert delta["improved_documents"] == delta["worsened_documents"] == 1
    assert delta["tied_documents"] == 1
    assert delta["largest_document_share_of_positive_gains"] == 1
    candidate["documents"].reverse()
    with pytest.raises(ValueError, match="unmatched"):
        report.paired(candidate, base)


def test_report_beam_and_greedy_histories(search: ModuleType) -> None:
    report = importlib.import_module("report_allocation_v2")
    cost = {
        "logical_unique_mask_document_evaluations": {"1024": 12},
        "physical_evaluation_seconds": 1.0,
        "completed_attempt_wall_seconds": 2.0,
    }
    shared = {
        "initial_nll": 2.0,
        "final_nll": 1.0,
        "final_layers": list(range(14)),
        "cost": cost,
    }
    nominations = {
        "all_seed_finals": {
            "greedy": {
                "odd": {
                    **shared,
                    "history": [
                        {
                            "iteration": 1,
                            "accepted": True,
                            "before_nll": 2.0,
                            "best_swap": {"mean_nll": 1.0},
                        },
                        {
                            "iteration": 2,
                            "accepted": False,
                            "before_nll": 1.0,
                            "best_swap": {"mean_nll": 1.1},
                        },
                    ],
                }
            },
            "beam": {
                "odd": {
                    **shared,
                    "history": [{"step": 1, "incumbent_nll": 1.0}],
                }
            },
        }
    }
    finals, steps = report.seed_rows(nominations)
    assert len(finals) == 2 and len(steps) == 5
    assert [s["nll"] for s in steps if s["method"] == "greedy"] == [2.0, 1.0, 1.0]
    assert finals[0]["logical_256_document_evaluations"] == 0


def test_device_staging_shares_only_new_experiment_part1(
    search: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = importlib.import_module("benchmark_allocation_v2")
    (tmp_path / "reports").mkdir()
    (tmp_path / "candidate").mkdir()
    bins = {f"part{i}_of_4.bin": str(i) * 64 for i in range(1, 5)}
    for group in ("candidate", "uniform_odd"):
        (tmp_path / "reports" / f"ready_{group}.json").write_text(
            json.dumps({"binaries_sha256": bins})
        )
    for name in bins:
        (tmp_path / "candidate" / name).write_bytes(b"fixture")
    prefix = "/data/local/tmp/qaihm_turboquant/llm/bundles/" + tmp_path.name
    remote = {prefix + "_uniform_odd/part1_of_4.bin": bins["part1_of_4.bin"]}
    calls = []

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        args = command[2:]
        calls.append(args)
        if args[0] == "sha256sum":
            text = remote.get(args[1], "")
        elif args[0] == "ln":
            assert args[1] == "-s"
            assert args[2].startswith(prefix + "_uniform_odd/")
            assert args[3].startswith(prefix + "_candidate/")
            remote[args[3]] = remote[args[2]]
            text = ""
        elif args[0] == "df":
            text = "Filesystem 1K-blocks Used Available Use% Mounted\n/data 9999999 0 9999999 0% /data"
        else:
            assert args[:2] == ["mkdir", "-p"]
            text = ""
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    monkeypatch.setattr(bench.subprocess, "run", execute)
    bench.stage_shared_part(tmp_path, "candidate")
    staged = json.loads((tmp_path / "reports/staging_candidate.json").read_text())
    assert staged["shared_part1_with_this_experiment_only"]
    assert staged["missing_binary_bytes"] == 3 * len(b"fixture")
    assert not staged["previous_device_artifacts_deleted"]
    assert not any(c[0] in ("rm", "mv", "cp") for c in calls)
    remote[prefix + "_candidate/part1_of_4.bin"] = "changed"
    with pytest.raises(AssertionError, match="Never overwrite"):
        bench.stage_shared_part(tmp_path, "candidate")


def test_prebuild_requires_completed_method_and_keeps_nomination_rule(
    search: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = importlib.import_module("benchmark_allocation_v2")
    for name in ("offline", "reports", "plans"):
        (tmp_path / name).mkdir()
    identity = tmp_path / "offline/scorer_identity.json"
    identity.write_text("{}")
    protocol = {"search": {"starts": ["uniform_odd", "uniform_even"]}}
    odd = {
        "run": "greedy_uniform_odd",
        "final_nll": 2.0,
        "final_layers": list(range(1, 28, 2)),
        "scorer_identity": bench.digest(identity),
        "cost": {"incomplete_attempts": 0},
    }
    even = {
        "run": "greedy_uniform_even",
        "final_nll": 2.0,
        "final_layers": list(range(0, 28, 2)),
        "scorer_identity": bench.digest(identity),
        "cost": {"incomplete_attempts": 0},
    }
    (tmp_path / "offline/greedy_uniform_odd.json").write_text(json.dumps(odd))
    built = []
    monkeypatch.setattr(bench, "ensure_built", lambda root, name, p: built.append(name))
    with pytest.raises(FileNotFoundError):
        bench.prebuild(tmp_path, "multi_greedy", protocol)
    assert not (tmp_path / "plans/multi_greedy.json").exists()
    bench.prebuild(tmp_path, "legacy_greedy", protocol)
    assert (
        json.loads((tmp_path / "plans/legacy_greedy.json").read_text())["k8_layers"]
        == odd["final_layers"]
    )
    path = tmp_path / "offline/greedy_uniform_even.json"
    path.write_text(json.dumps(even))
    bench.prebuild(tmp_path, "multi_greedy", protocol)
    plan = json.loads((tmp_path / "plans/multi_greedy.json").read_text())
    assert plan["k8_layers"] == even["final_layers"]  # Ascending mask breaks NLL tie.
    (tmp_path / "reports/ready_alias.json").write_text(
        json.dumps({"config_hash": plan["config_hash"]})
    )
    bench.prebuild(tmp_path, "multi_greedy", protocol)
    assert built == ["legacy_greedy", "multi_greedy"]
    even["final_nll"] = 1.9
    path.write_text(json.dumps(even))
    with pytest.raises(AssertionError):
        bench.prebuild(tmp_path, "multi_greedy", protocol)


def test_cache_identity_length_and_logical_cost(
    search: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "offline").mkdir()
    (tmp_path / "assets").mkdir()
    windows = {}
    for role, count in (("search", 12), ("validation", 4)):
        rows = []
        for i in range(count):
            name = f"{role}_{i}.bin"
            (
                np.arange(1024, dtype="<i4")
                + i * 2000
                + (100000 if role == "validation" else 0)
            ).tofile(tmp_path / "assets" / name)
            rows.append({"file": name, "title": f"= {role} {i} ="})
        windows[role] = rows
    (tmp_path / "data_manifest.json").write_text(json.dumps({"windows": windows}))

    class FakeReplay:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, tokens: np.ndarray, layers: list[int]) -> dict:
            self.calls += 1
            count = len(tokens) - 1
            return {
                "nll_sum": float(sum(layers) + int(tokens[0])) * count,
                "scored_tokens": count,
                "elapsed_s": 0.01,
            }

    coarse, full = FakeReplay(), FakeReplay()
    cache = search.Scores(tmp_path, {256: coarse, 1024: full}, {"numeric": 1})
    mask = tuple(range(1, 28, 2))
    try:
        cache.run = "first"
        value = cache(mask, "coarse")
        assert value == cache(mask, "coarse") and coarse.calls == 2
        cache(mask)
        assert full.calls == 12
        cache.run = "second"
        cache(mask)
        assert full.calls == 12
        stats = cache.cost(["second"])
        assert stats["logical_unique_mask_document_evaluations"] == {"1024": 12}
        assert stats["physical_cache_misses"] == {}
        cache(list(mask[:-1]))  # Legacy-only, nondeployable 13-layer diagnostic.
        assert full.calls == 24
        with pytest.raises(ValueError, match="independent test"):
            cache(mask, "test")
        with pytest.raises(ValueError, match="differs"):
            search.Scores(tmp_path, {256: coarse, 1024: full}, {"numeric": 2})
        report = importlib.import_module("report_allocation_v2")
        rows = report.offline_documents(tmp_path)
        assert len(rows) == 26 and sum(r["diagnostic_only"] for r in rows) == 12
        assert all(
            r["split"] == "search" and not r["title"].startswith("=") for r in rows
        )
        assert all(
            r["delta_vs_odd_if_measured"] == 0 for r in rows if r["k8_count"] == 14
        )
        work = report.mask_workload(tmp_path, ["second"])
        assert work["1024"]["unique_14_layer_masks"] == 1
        assert work["1024"]["unique_13_or_15_layer_diagnostic_masks"] == 1
    finally:
        cache.db.close()


def test_prepared_structured_constants_preserve_eager_codec(search: ModuleType) -> None:
    model = importlib.import_module("allocation_offline_model")
    x = torch.randn(2, 3, 128, generator=torch.Generator().manual_seed(852))
    params = torch.tensor(model.coefficients().astype(np.float16).astype(np.float32))
    torch.testing.assert_close(
        model.codec(x, 8, 42, True),
        model.codec(x, 8, 42, True, structured_params=params),
        atol=0,
        rtol=0,
    )


def test_final_report_aggregates_tokens_and_deduplicates_aliases(
    search: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = importlib.import_module("report_allocation_v2")
    for name in ("assets", "reports", "offline", "plans"):
        (tmp_path / name).mkdir()

    def save(name: str, value: dict) -> None:
        (tmp_path / name).write_text(json.dumps(value))

    aliases = {
        "uniform_odd": "uniform_odd",
        "early": "uniform_odd",
        "boundary": "candidate",
        "legacy_greedy": "candidate",
        "multi_greedy": "candidate",
        "multi_beam": "candidate",
        "k6_v2": "k6_v2",
    }
    groups = ["uniform_odd", "candidate", "k6_v2"]
    sha = {f"test_{i}.bin": f"test-{i}" for i in range(8)}
    sha.update({f"legacy_{i}.bin": f"legacy-{i}" for i in range(4)})
    save(
        "protocol.json",
        {
            "assets_manifest": {
                "sha256": sha,
                "wikitext_windows": [f"legacy_{i}.bin" for i in range(4)],
            },
            "search": {"starts": ["uniform_odd"]},
            "evaluation": {"groups": list(aliases)},
        },
    )
    save(
        "data_manifest.json",
        {
            "windows": {
                "search": [{"file": "search.bin", "title": "= Search ="}],
                "validation": [],
                "test": [
                    {"file": f"test_{i}.bin", "title": f"= Doc {i} ="} for i in range(8)
                ],
            }
        },
    )
    np.arange(1024, dtype="<i4").tofile(tmp_path / "assets/search.bin")

    class FakeReplay:
        def score(self, tokens: np.ndarray, layers: list[int]) -> dict:
            return {"nll_sum": 2046.0, "scored_tokens": 1023, "elapsed_s": 0.01}

    cache = search.Scores(tmp_path, {1024: FakeReplay()}, {})
    cache.run = "greedy_uniform_odd"
    cache(tuple(range(1, 28, 2)))
    cost = cache.cost([cache.run])
    cache.db.close()
    seed = {
        "initial_nll": 2.0,
        "final_nll": 1.9,
        "final_layers": list(range(14)),
        "history": [],
        "cost": cost,
    }
    save(
        "nominations.json",
        {
            "all_seed_finals": {m: {"uniform_odd": seed} for m in ("greedy", "beam")},
            "method_costs": {},
        },
    )
    save("physical_groups.json", {"groups": groups, "aliases": aliases})
    save("selection.json", {"selected_group": "candidate", "htp_mean_nll": {}})
    save(
        "local_certificate.json",
        {
            "selection_sha256": report.digest(tmp_path / "selection.json"),
            "layers": list(range(14)),
            "swaps": [
                {"promote": p, "demote": d, "delta_nll": (p - d - 14) / 1000}
                for p in range(14, 28)
                for d in range(14)
            ],
        },
    )
    for group, base in zip(groups, (2.0, 1.9, 1.8), strict=True):
        (tmp_path / group).mkdir()
        save(f"{group}/convert_report.json", {"config_hash": group})
        save(f"plans/{group}.json", {"k8_layers": list(range(14))})
        for part in range(1, 5):
            (tmp_path / group / f"part{part}_of_4.bin").write_bytes(b"fixture")
        for role, count in (("test", 8), ("legacy", 4)):
            for i in range(count):
                # Distinct document NLLs detect mean(PPL) versus exp(mean(NLL)).
                nll = base + i / 10
                save(
                    f"reports/{group}_{role}_{i:02d}.json",
                    {
                        "config_hash": group,
                        "assets": {"tokens_sha256": sha[f"{role}_{i}.bin"]},
                        "scored_tokens": 1023,
                        "kv_store_bytes": 30277632,
                        "backend_api": "2.37.0/htp 5.48.0",
                        "device": {"soc_model": "SM8850"},
                        "nll_sum": nll * 1023,
                        "ppl": float(np.exp(nll)),
                    },
                )

    def perf(path: Path, prompt: int) -> dict:
        assert prompt == 897
        value = float(path.stem[-1])
        return {
            "host_kv_MiB": 28.875,
            **dict.fromkeys(
                (
                    "ttft_ms",
                    "prefill_tok_per_s",
                    "decode_tok_per_s",
                    "io_buffer_MiB",
                    "end_VmRSS_MiB",
                    "process_VmHWM_MiB",
                    "mean_decode_prepare_ms",
                    "mean_decode_qnn_ms",
                    "mean_decode_commit_ms",
                ),
                value,
            ),
        }

    monkeypatch.setattr(report, "performance", perf)
    monkeypatch.setattr(report, "figures", lambda *args: None)
    report.summary(tmp_path)
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert len(result["rows"]) == 7 and len(result["groups"]) == 3
    assert result["performance_sessions"] == 9
    quality = result["groups"]["candidate"]["quality"]["test"]
    assert quality["scored_tokens"] == 8 * 1023
    assert quality["mean_nll"] == pytest.approx(2.25)
    assert quality["ppl"] == pytest.approx(float(np.exp(2.25)))
    assert (
        result["test_contrasts"]["multi_beam"]["uniform_odd"]["improved_documents"] == 8
    )
    assert result["test_contrasts"]["multi_beam"]["k6_v2"]["worsened_documents"] == 8
    assert result["test_contrasts"]["multi_beam"]["multi_greedy"]["tied_documents"] == 8
    assert result["groups"]["candidate"]["performance"]["ttft_ms"]["std_ddof1"] == 1
    assert len((tmp_path / "document_nll.csv").read_text().splitlines()) == 1 + 3 * 12
    assert (tmp_path / "offline_document_nll.csv").is_file()
    snapshot = result["score_cache_snapshot"]
    assert snapshot["rows"]["scores"] == 1
    assert report.digest(Path(snapshot["path"])) == snapshot["sha256"]
    with pytest.raises(FileExistsError):
        report.snapshot_cache(tmp_path)


def test_snapshot_retains_committed_wal_rows(
    search: ModuleType, tmp_path: Path
) -> None:
    report = importlib.import_module("report_allocation_v2")
    (tmp_path / "offline").mkdir()
    live = tmp_path / "offline/scores.sqlite"
    writer = sqlite3.connect(live)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        for table in ("scores", "requests", "attempts"):
            writer.execute(f"CREATE TABLE {table}(value INTEGER)")
        writer.execute("INSERT INTO scores VALUES(1),(2)")
        writer.execute("INSERT INTO requests VALUES(3)")
        writer.commit()
        wal = Path(str(live) + "-wal")
        assert wal.stat().st_size > 0
        before = (report.digest(live), report.digest(wal))
        snapshot = report.snapshot_cache(tmp_path)
        assert snapshot["rows"] == {"scores": 2, "requests": 1, "attempts": 0}
        assert before == (report.digest(live), report.digest(wal))
        with sqlite3.connect(
            f"file:{snapshot['path']}?mode=ro&immutable=1", uri=True
        ) as restored:
            assert restored.execute("SELECT * FROM scores").fetchall() == [(1,), (2,)]
            assert restored.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        writer.close()


def test_device_score_helper_requires_frozen_selection_for_test(
    search: ModuleType, tmp_path: Path
) -> None:
    benchmark = importlib.import_module("benchmark_allocation_v2")
    for role in ("test", "legacy"):
        with pytest.raises(FileNotFoundError, match=r"selection\.json"):
            benchmark.score_set(tmp_path, "candidate", role, {})
    for role in ("search", "validation"):
        with pytest.raises(FileNotFoundError, match=r"data_manifest\.json"):
            benchmark.score_set(tmp_path, "candidate", role, {})
    (tmp_path / "selection.json").write_text(json.dumps({"test_used": True}))
    with pytest.raises(ValueError, match="must not use independent test"):
        benchmark.score_set(tmp_path, "candidate", "test", {})
    with pytest.raises(ValueError, match="Unknown evaluation split"):
        benchmark.score_set(tmp_path, "candidate", "unregistered", {})


def test_optional_report_figures(search: ModuleType, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    report = importlib.import_module("report_allocation_v2")
    (tmp_path / "protocol.json").write_text(
        json.dumps({"search": {"starts": ["example"]}})
    )
    (tmp_path / "local_certificate.json").write_text(
        json.dumps(
            {
                "layers": list(range(14)),
                "swaps": [
                    {"promote": p, "demote": d, "delta_nll": (p - d - 14) / 1000}
                    for p in range(14, 28)
                    for d in range(14)
                ],
            }
        )
    )
    report.figures(
        tmp_path,
        {"example": {"k8_layers": list(range(14))}, "k6_v2": {"k8_layers": None}},
        [
            {"method": m, "start": "example", "step": 0, "nll": 3.0}
            for m in ("greedy", "beam")
        ],
        "example",
    )
    for name in ("layer_masks.svg", "search_history.svg", "local_swap_delta.svg"):
        assert ET.parse(tmp_path / "figures" / name).getroot().tag.endswith("svg")


def test_orchestration_import_does_not_require_tokenizer(search: ModuleType) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name == "transformers" or name.startswith("transformers."):
        raise ImportError("Tokenizer dependency is not available")
    return original(name, *args, **kwargs)
builtins.__import__ = checked
sys.path.insert(0, sys.argv[1])
import benchmark_allocation_v2
import report_allocation_v2
import verify_allocation_v2
""",
            str(scripts),
        ],
        check=True,
        timeout=60,
    )


def test_screen_diagnostic_uses_only_unvisited_excluded_neighbors(
    search: ModuleType,
) -> None:
    report = importlib.import_module("report_allocation_v2")
    initial = list(range(14))
    neighbors = [
        {**r, "coarse_nll": float(i)}
        for i, r in enumerate(search.swaps(tuple(initial)))
    ]
    selected = [r["layers"] for r in neighbors[:4]]
    excluded = neighbors[4]["layers"]
    result = {
        "initial_layers": initial,
        "initial_nll": 3.5,
        "history": [
            {
                "step": 1,
                "parents": [{"all_196_swaps": neighbors, "full_shortlist": selected}],
                "full_evaluated": [{"layers": m, "mean_nll": 3.0} for m in selected],
                "incumbent_nll": 3.0,
            }
        ],
    }
    nominations = {"all_seed_finals": {"beam": {"example": result}}}
    diagnostic = report.screening_diagnostics(
        nominations, {excluded: 2.8, selected[0]: 2.0}
    )
    assert diagnostic["steps_with_missed_improvement"] == 1
    row = diagnostic["rows"][0]
    assert row["excluded_neighbors_with_prior_greedy_full_scores"] == 1
    assert row["coarse_ranks_among_unvisited_neighbors"] == [5]
    assert row["additional_search_nll_reduction_missed"] == pytest.approx(0.2)
    assert (
        report.screening_diagnostics(nominations, {excluded: 3.1})[
            "steps_with_missed_improvement"
        ]
        == 0
    )
