# SPDX-License-Identifier: BSD-3-Clause
"""NIAH build/provenance contracts without QNN conversion or device execution."""

import copy
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def niah(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_build")


def test_estimate_matches_observed_1024_buffers(niah: ModuleType) -> None:
    fp16 = niah.memory_estimate(1024, False)
    compressed = niah.memory_estimate(1024, True)
    assert fp16["host_kv_bytes"] == 117440512
    assert fp16["runtime_io_bytes"] == 275963396
    assert compressed["host_kv_bytes"] == 30277632
    assert compressed["runtime_io_bytes"] == 101637636
    assert niah.memory_estimate(32768, False)["explicit_host_bytes"] == 11323561476
    assert niah.memory_estimate(16384, True)["host_kv_mib"] == 462
    for invalid in (0, 128, -1, True, 1024.0):
        with pytest.raises(ValueError, match="integer context"):
            niah.memory_estimate(invalid, True)


def test_profiles_and_packages_preserve_historical_selection(niah: ModuleType) -> None:
    assert niah.CAPACITIES == (1024, 4096, 8192, 16384)
    assert niah.config("fp16").profile == "baseline_fp16_kv_fp16_attn"
    assert niah.native("fp16") is None
    assert niah.native("k4_v4").name == "native_decoder_hvx_20260918"
    assert niah.native("boundary_lm") == niah.native("boundary_structured")
    assert (
        niah.config("k4_v4").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    lm = niah.config("boundary_lm")
    structured = niah.config("boundary_structured")
    assert lm.layer_bits == structured.layer_bits
    assert [i for i, (k, _) in enumerate(lm.layer_bits) if k == 8] == list(
        range(7)
    ) + list(range(21, 28))
    assert all(v == 2 for _, v in lm.layer_bits)
    assert lm.converged_k8 and structured.structured_k8
    with pytest.raises(ValueError, match="odd_lm"):
        niah.config("odd_lm")


def test_build_options_exact_scope(niah: ModuleType, tmp_path: Path) -> None:
    for group in niah.GROUPS:
        options = niah.convert_options(tmp_path, 4096, group, 2, tmp_path / "out")
        assert options[options.index("--context-length") + 1] == "4096"
        assert options[options.index("--parts") + 1] == "2"
        assert options[
            options.index("--sequence-lengths") + 1 : options.index(
                "--sequence-lengths"
            )
            + 3
        ] == ["1", "128"]
        assert ("--structured-k8-decoder" in options) == (
            group == "boundary_structured"
        )
        assert ("--layer-plan" in options) == (group == "boundary_structured")
        assert ("--native-decoder-package" in options) == (group != "fp16")


def test_staging_metadata_private_and_original_immutable(
    niah: ModuleType, tmp_path: Path
) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "part1_of_4.bin").write_bytes(b"original")
    metadata = {"profile": "test", "parts": {}}
    (source / "convert_report.json").write_text(json.dumps(metadata))
    niah.stage(source, target)
    assert (target / "part1_of_4.bin").is_symlink()
    assert not (target / "convert_report.json").is_symlink()
    (target / "convert_report.json").write_text("{}")
    assert json.loads((source / "convert_report.json").read_text()) == metadata
    with pytest.raises(FileExistsError):
        niah.stage(source, target)


def test_unauthorized_capacity_rejected_before_source_work(
    niah: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "protocol.json").write_text(
        json.dumps({"context_capacities": [1024, 4096, 8192, 16384]})
    )
    with pytest.raises(ValueError, match="not authorized"):
        niah.check(tmp_path, 32768)


def test_incomplete_or_wrong_context_not_reused(
    niah: ModuleType, tmp_path: Path
) -> None:
    assert not niah.completed_part(tmp_path, 2, 4096, "fp16")
    data = {
        "config_hash": niah.config("fp16").config_hash(),
        "context_buckets": [4096],
        "parts": {"part2_of_4": {"graphs": {}}},
    }
    (tmp_path / "convert_report.json").write_text(json.dumps(data))
    assert not niah.completed_part(tmp_path, 2, 4096, "fp16")
    data["parts"]["part2_of_4"]["context_s"] = 1
    (tmp_path / "convert_report.json").write_text(json.dumps(data))
    (tmp_path / "part2_of_4.bin").write_bytes(b"binary")
    assert niah.completed_part(tmp_path, 2, 4096, "fp16")
    with pytest.raises(ValueError, match="Incompatible completed"):
        niah.completed_part(tmp_path, 2, 8192, "fp16")


def fake_contexts(capacity: int, config: object) -> list[dict]:
    graphs = []
    for ar in (1, 128):
        inputs, outputs = [], []

        def tensor(
            name: str, dtype: str, shape: list[int], quantized: bool = False
        ) -> dict:
            return {
                "info": {
                    "name": name,
                    "dataType": dtype,
                    "dimensions": shape,
                    "quantizeParams": {
                        "definition": "QNN_DEFINITION_DEFINED"
                        if quantized
                        else "QNN_DEFINITION_UNDEFINED"
                    },
                }
            }

        for name, shape, dtype in (
            ("input_ids", [1, ar], "QNN_DATATYPE_INT_32"),
            ("embedding", [1, ar, 2048], "QNN_DATATYPE_UFIXED_POINT_16"),
            ("add_21844", [1, ar, 2048], "QNN_DATATYPE_UFIXED_POINT_16"),
            ("add_42314", [1, ar, 2048], "QNN_DATATYPE_UFIXED_POINT_16"),
            ("attention_mask", [1, 1, ar, capacity], "QNN_DATATYPE_UFIXED_POINT_16"),
            ("position_ids_cos", [1, 1, ar, 64], "QNN_DATATYPE_UFIXED_POINT_16"),
            ("position_ids_sin", [1, 1, ar, 64], "QNN_DATATYPE_UFIXED_POINT_16"),
            ("logits", [1, ar, 151936], "QNN_DATATYPE_UFIXED_POINT_16"),
        ):
            inputs.append(tensor(name, dtype, shape, "UFIXED" in dtype))
        for layer in range(28):
            for kind in ("key", "value"):
                for side, count, dest in (
                    ("in", capacity - ar, inputs),
                    ("out", ar, outputs),
                ):
                    if config.enabled:
                        for field, width, dtype in (
                            (
                                "packed",
                                16 * config.codec_for(kind, layer).bits,
                                "QNN_DATATYPE_UINT_8",
                            ),
                            ("scale", 1, "QNN_DATATYPE_FLOAT_16"),
                        ):
                            dest.append(
                                tensor(
                                    f"tq_{kind}_{layer}_{field}_{side}",
                                    dtype,
                                    [8, 1, count, width],
                                )
                            )
                    else:
                        shape = (
                            [8, 1, 128, count] if kind == "key" else [8, 1, count, 128]
                        )
                        dest.append(
                            tensor(
                                f"past_{kind}_{layer}_{side}",
                                "QNN_DATATYPE_FLOAT_16",
                                shape,
                            )
                        )
        graphs.append(
            {
                "info": {
                    "graphName": f"{'token' if ar == 1 else 'prompt'}_ar{ar}_cl{capacity}_1_of_4",
                    "graphInputs": inputs,
                    "graphOutputs": outputs,
                }
            }
        )
    return [{"info": {"graphs": graphs}}]


@pytest.mark.parametrize(
    "group", ["fp16", "k4_v4", "boundary_lm", "boundary_structured"]
)
def test_compiled_contract_stream_shape_dtype_and_total_memory(
    niah: ModuleType, group: str
) -> None:
    c = niah.config(group)
    contexts = fake_contexts(4096, c)
    good = niah.compiled_io_contract(contexts, 4096, c)
    assert good["passed"], good["violations"]
    assert (
        good["runtime_io_bytes"]
        == niah.memory_estimate(4096, group != "fp16")["runtime_io_bytes"]
    )
    changed = copy.deepcopy(contexts)
    changed[0]["info"]["graphs"][0]["info"]["graphInputs"][-1]["info"]["dimensions"][
        2
    ] = 7
    assert not niah.compiled_io_contract(changed, 4096, c)["passed"]
    changed = copy.deepcopy(contexts)
    changed[0]["info"]["graphs"][0]["info"]["graphInputs"][-1]["info"]["dataType"] = (
        "QNN_DATATYPE_UFIXED_POINT_8"
    )
    assert not niah.compiled_io_contract(changed, 4096, c)["passed"]
    assert not niah.compiled_io_contract(contexts, 8192, c)["passed"]
