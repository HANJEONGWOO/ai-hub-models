# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Fail-closed activation-only calibration policy and FP16 simulation tests."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

from qai_hub_models.models.templates.llm.turboquant.calibration import (
    assert_fp16_boundaries,
    create_sim,
    digest,
    finish_observers,
    load_calibrated,
    parameter_interfaces,
    preserve_parameter_interfaces,
    range_session,
    start_observers,
    validate_encodings,
)
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.fp16_attention import (
    use_fp16_kv_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    SurgeryResult,
    apply_kv_profile,
)
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    CONTEXT,
    attention_part,
)


def target(seq: int = 1) -> SurgeryResult:
    model, enc = attention_part(seq)
    config = get_profile("baseline_fp16_kv_fp16_attn_calibrated")
    return use_fp16_kv_attention(
        apply_kv_profile(model, enc, config, seq, CONTEXT), config
    )


def test_legacy_hash_and_opt_in_profile() -> None:
    old = get_profile("baseline_fp16_kv_fp16_attn")
    new = get_profile("baseline_fp16_kv_fp16_attn_calibrated")
    assert (
        old.config_hash()
        == "359d76b2046409c544568a5de39980bece960e7c1c966480fee1e9ad3898b94b"
    )
    assert not old.activation_calibrated
    assert new.activation_calibrated and new.fp16_attention and not new.enabled
    assert new.config_hash() != old.config_hash()
    assert new.to_dict()["attention"]["recalibrated"]


@pytest.mark.parametrize(
    "fault", ["weight", "boundary", "bits", "symmetry", "nan", "zero", "offset"]
)
def test_encoding_validation_rejects_mutations(fault: str) -> None:
    old = target().encodings
    new = copy.deepcopy(old)
    enc = new["activation_encodings"][0]
    if fault == "weight":
        new["param_encodings"].append({"name": "changed"})
    elif fault == "boundary":
        new["activation_encodings"].pop()
    elif fault == "bits":
        enc["bw"] = 8
    elif fault == "symmetry":
        enc["is_sym"] = not enc["is_sym"]
    elif fault == "nan":
        enc["scale"] = [float("nan")]
    elif fault == "zero":
        enc["scale"] = [0.0]
    else:
        enc["offset"] = [-0.5]
    with pytest.raises(ValueError, match=r"Calibration|Invalid"):
        validate_encodings(old, new)


def test_fp16_tensor_cannot_acquire_integer_encoding() -> None:
    result = target()
    enc = copy.deepcopy(result.encodings)
    enc["activation_encodings"].append(
        {**enc["activation_encodings"][0], "name": "past_key_0_in"}
    )
    with pytest.raises(ValueError, match="FP16"):
        assert_fp16_boundaries(result.model, enc)


@pytest.mark.parametrize(
    "fault", [None, "incomplete", "profile", "source", "weights", "encodings"]
)
def test_calibrated_artifact_identity(tmp_path: Path, fault: str | None) -> None:
    source = tmp_path / "part.onnx"
    weights = tmp_path / "part.data"
    source.write_bytes(b"original graph")
    weights.write_bytes(b"original weights")
    original = target().encodings
    source.with_suffix(".encodings").write_text(json.dumps(original))
    calibrated = copy.deepcopy(original)
    calibrated["activation_encodings"][0]["scale"][0] *= 2
    artifact = tmp_path / "calibrated.encodings"
    artifact.write_text(json.dumps(calibrated))
    manifest = {
        "status": "complete",
        "profile": "baseline_fp16_kv_fp16_attn_calibrated",
        "parts": {
            "part": {
                "source_onnx_sha256": digest(source),
                "source_encodings_sha256": digest(source.with_suffix(".encodings")),
                "source_data_sha256": {weights.name: digest(weights)},
                "encodings_file": artifact.name,
                "encodings_sha256": digest(artifact),
            }
        },
    }
    if fault == "incomplete":
        manifest["status"] = "partial"
    elif fault == "profile":
        manifest["profile"] = "baseline_fp16_kv_fp16_attn"
    elif fault in ("source", "weights", "encodings"):
        {"source": source, "weights": weights, "encodings": artifact}[
            fault
        ].write_bytes(b"modified")
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest))
    if fault is None:
        actual, report = load_calibrated(tmp_path, source, original)
        assert actual == calibrated
        assert report["changed_count"] == 1
    else:
        with pytest.raises(ValueError, match=r"Incomplete|Calibration|Calibrated"):
            load_calibrated(tmp_path, source, original)


def test_cpu_observation_keeps_cache_half_and_updates_only_ranges(
    tmp_path: Path,
) -> None:
    pytest.importorskip("aimet_onnx")
    result = target()
    # Non-linear parameters (RMSNorm-style Mul weights) are not folded by
    # AIMET's fold_param_quantizers; their frozen QDQ must remain in observers.
    for value in result.model.graph.input:
        if value.name == "new_key":
            value.name = "raw_key"
    result.model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.asarray([1.234], dtype=np.float32), name="norm_weight"
        )
    )
    result.model.graph.node.insert(
        0,
        onnx.helper.make_node(
            "Mul", ["raw_key", "norm_weight"], ["new_key"], name="norm_mul"
        ),
    )
    result.encodings["param_encodings"].append(
        {
            "name": "norm_weight",
            "bw": 16,
            "dtype": "INT",
            "enc_type": "PER_TENSOR",
            "is_sym": False,
            "scale": [0.1],
            "offset": [-32768.0],
        }
    )
    original = copy.deepcopy(result.encodings)
    rng = np.random.default_rng(12)
    data = {
        v.name: (
            rng.standard_normal([d.dim_value for d in v.type.tensor_type.shape.dim])
            * 0.2
        ).astype(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))
        for v in result.model.graph.input
    }
    sim = create_sim(
        result.model,
        result.encodings,
        data,
        tmp_path,
        providers=["CPUExecutionProvider"],
    )
    start_observers(sim, result.encodings)
    output = dict(
        zip(
            [o.name for o in sim.session.get_outputs()],
            sim.session.run(None, data),
            strict=True,
        )
    )
    assert all(
        v.dtype == np.float16 for n, v in output.items() if n.startswith("past_")
    )
    encodings, report = finish_observers(sim, result.encodings)
    assert report["changed_count"] > 0
    assert result.encodings == original
    assert encodings["param_encodings"] == original["param_encodings"]
    validate_encodings(original, encodings)
    observer = range_session(sim, original, tmp_path / "native_observers")
    ranges = observer.run(["calibration_ranges"], data)[0].reshape(-1, 2)
    start_observers(sim, original)
    for e, bounds in zip(original["activation_encodings"], ranges, strict=True):
        sim.qc_quantize_op_dict[e["name"]].update_encoding_stats(bounds)
    native_encodings, _ = finish_observers(sim, original)
    for a, b in zip(
        encodings["activation_encodings"],
        native_encodings["activation_encodings"],
        strict=True,
    ):
        np.testing.assert_allclose(a["scale"], b["scale"], rtol=1e-6)
        np.testing.assert_array_equal(a["offset"], b["offset"])


def test_schedule_covers_prefill_and_nonempty_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("calibrate_fp16_attention")
    steps = list(module.schedule())
    assert sum(s[3] for s in steps if s[0] == "prefill") == 1023
    assert [s[2] for s in steps if s[0] == "decode"] == [
        128,
        256,
        384,
        512,
        640,
        768,
        896,
        1023,
    ]
    assert steps[-1] == ("decode", 7, 1023, 1, 1)


@pytest.mark.parametrize("fault", [None, "missing", "different"])
def test_calibrated_partial_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    assembler = importlib.import_module("assemble_native_bundle")
    sources = [tmp_path / "first", tmp_path / "second"]
    for number, path in enumerate(sources, 1):
        path.mkdir()
        name = f"part{number}_of_2"
        report = {
            "profile": "baseline_fp16_kv_fp16_attn_calibrated",
            "config_hash": "calibrated",
            "context_length": 1024,
            "context_buckets": [1024],
            "attention_tile": 0,
            "rotated_attention": False,
            "native_decoder": None,
            "activation_calibration_sha256": "same-calibration",
            "parts": {
                name: {
                    "context_s": 1,
                    "graphs": {f"token_ar1_cl1024_{number}_of_2": {}},
                }
            },
        }
        if fault == "missing":
            del report["activation_calibration_sha256"]
        elif fault == "different" and number == 2:
            report["activation_calibration_sha256"] = "different-calibration"
        (path / "convert_report.json").write_text(json.dumps(report))
        (path / f"{name}.bin").write_bytes(b"test context")
    output = tmp_path / "bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble",
            "--base",
            str(sources[0]),
            "--parts",
            str(sources[1]),
            "--out",
            str(output),
            "--fp16-attention",
        ],
    )
    if fault:
        with pytest.raises(ValueError, match=r"identity|Incompatible"):
            assembler.main()
        assert not output.exists()
    else:
        assembler.main()
        report = json.loads((output / "convert_report.json").read_text())
        assert report["activation_calibration_sha256"] == "same-calibration"
        assert len(report["parts"]) == 2


def test_gather_parameter_grid_is_not_independently_recalibrated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "embedding.onnx"
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Gather", ["table", "ids"], ["embedding"])],
            "embedding",
            [onnx.helper.make_tensor_value_info("ids", onnx.TensorProto.INT32, [1])],
            [
                onnx.helper.make_tensor_value_info(
                    "embedding", onnx.TensorProto.FLOAT, [1]
                )
            ],
        )
    )
    onnx.save(model, source)
    grid = {**target().encodings["activation_encodings"][0], "name": "table"}
    source.with_suffix(".encodings").write_text(
        json.dumps({"param_encodings": [grid], "activation_encodings": []})
    )
    fixed = parameter_interfaces([source])
    original = {"param_encodings": [], "activation_encodings": [fixed["embedding"]]}
    observed = copy.deepcopy(original)
    observed["activation_encodings"][0]["scale"][0] *= 0.9
    restored, kept = preserve_parameter_interfaces(original, observed, fixed)
    assert restored == original
    assert kept == ["embedding"]
    assert observed != restored


@pytest.mark.parametrize("mismatch", [False, True])
def test_compiled_shared_interface_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: bool
) -> None:
    from types import SimpleNamespace

    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    audit = importlib.import_module("verify_fp16_attention")
    row = {"dims": "1,1,2048", "dtype": "uFxp_16", "encoding": "scale: fixed"}
    producer = SimpleNamespace(io_tables={"input": {}, "output": {"embedding": row}})
    consumer = SimpleNamespace(
        io_tables={
            "input": {
                "embedding": {**row, "encoding": "scale: different"}
                if mismatch
                else row
            },
            "output": {},
        }
    )
    monkeypatch.setattr(
        audit, "parse_dlcinfo", lambda p: producer if "_1_of_2" in p.name else consumer
    )
    conversion = {
        "parts": {
            f"part{i}_of_2": {"graphs": {f"token_ar1_cl1024_{i}_of_2": {}}}
            for i in (1, 2)
        }
    }
    result = audit.verify_shared_interfaces(tmp_path, conversion)
    assert result["graph_sets"] == 1
    assert result["shared_tensors_checked"] == 1
    assert bool(result["violations"]) == mismatch
