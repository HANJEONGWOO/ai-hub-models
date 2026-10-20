# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Converged K8 changes constants only; byte format and legacy profiles stay fixed."""

import hashlib
import importlib
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.export import (
    Subgraph,
    _scalar_index_tree,
    build_encode_model,
)
from qai_hub_models.models.templates.llm.turboquant.lloyd_max_k8 import (
    codebook,
    metadata,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import native_table
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    FLOAT32_GRAPH,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
    spec_codebook,
)


def test_constants_and_hashes() -> None:
    c = codebook()
    assert c.shape == (256,) and c.dtype == np.float64
    assert np.all(np.diff(c) > 0) and np.array_equal(c, -c[::-1])
    assert np.all(np.diff(c.astype(np.float16)) > 0)
    b = (c[:-1] + c[1:]) / 2
    for label, array, dtype in (
        ("codebook_sha256", c, "<f8"),
        ("codebook_fp16_sha256", c, "<f2"),
        ("boundary_fp32_sha256", b, "<f4"),
        ("boundary_fp16_sha256", b, "<f2"),
    ):
        assert (
            hashlib.sha256(array.astype(dtype).tobytes()).hexdigest()
            == metadata()[label]
        )


def test_tree_edges_extremes_and_bytes() -> None:
    c = codebook()
    b = ((c[:-1] + c[1:]) / 2).astype(np.float32)
    x = np.sort(
        np.r_[
            b,
            np.nextafter(b, np.float32(-np.inf)),
            np.nextafter(b, np.float32(np.inf)),
            -1e6,
            0,
            1e6,
        ]
    ).astype(np.float32)
    sg = Subgraph()
    out = _scalar_index_tree(sg, "x", 8, 128, "test_", c)
    model = helper.make_model(
        helper.make_graph(
            sg.nodes,
            "tree",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
            [helper.make_tensor_value_info(out, TensorProto.INT32, list(x.shape))],
            list(sg.initializers.values()),
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    result = session.run(None, {"x": x})[0]
    np.testing.assert_array_equal(result, np.searchsorted(b, x, side="left"))
    assert result.min() == 0 and result.max() == 255
    codes = np.arange(256, dtype=np.uint8).reshape(2, 128)
    np.testing.assert_array_equal(pack_indices(codes, 8), codes)
    np.testing.assert_array_equal(unpack_indices(codes, 8, 128), codes)


def test_only_k8_changes_legacy_hash_budget_cache_rejection() -> None:
    old = get_profile("mixed_k6_v2_uniform")
    new = get_profile("mixed_k6_v2_uniform_lm8_converged")
    assert (
        old.config_hash()
        == "0c5924e389479226063bb62e5e812dcf406d25695ac9dcb28a0405bb567f1da4"
    )
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    assert (
        get_profile("mixed_k6_v2_uniform_structured_k8").config_hash()
        == "b68d0d5534b516363c892ecf65bd98f44789969d4e07cc31de73e7bdc3d11ed9"
    )
    for layer in range(28):
        for kind in ("key", "value"):
            a, b = [config.codec_for(kind, layer) for config in (old, new)]
            if kind == "value" or layer % 2 == 0:
                assert a == b
            else:
                assert b.codebook_variant and b.quantizer == a.quantizer == "lloyd_max"
                np.testing.assert_array_equal(spec_codebook(b, 128), codebook())
    caches = [TurboQuantKVCache(config, 28, 8, 128, 1024) for config in (old, new)]
    assert all(c.memory_report().allocated_bytes == 30277632 for c in caches)
    with pytest.raises(ValueError, match="different TurboQuant config"):
        caches[1].load_state_dict(caches[0].state_dict())
    with pytest.raises(ValueError, match="Converged K8 requires"):
        replace(new, structured_k8=True)


@pytest.mark.parametrize("tokens", [1, 3, 128])
def test_encoder_reference_and_unchanged_structure(tokens: int) -> None:
    configs = [get_profile(p) for p in ("k8_v4_scaled", "lm8_converged_v4_scaled")]
    models = [build_encode_model(c, c.key, 8, tokens) for c in configs]
    assert [n.SerializeToString() for n in models[0].graph.node] == [
        n.SerializeToString() for n in models[1].graph.node
    ]
    tables = [
        {t.name: numpy_helper.to_array(t) for t in m.graph.initializer} for m in models
    ]
    assert tables[0].keys() == tables[1].keys()
    changed = [k for k in tables[0] if not np.array_equal(tables[0][k], tables[1][k])]
    assert changed and all(
        k.startswith(("tq_scalar_b8_", "tq_centroid_b8_")) for k in changed
    )
    c = configs[1]
    assert native_table(c, "key") == "tq_native_converged_k8_centroids_fp16"
    x = (
        np.random.default_rng(tokens)
        .normal(size=(1, 8, tokens, 128))
        .astype(np.float32)
    )
    session = ort.InferenceSession(
        models[1].SerializeToString(), providers=["CPUExecutionProvider"]
    )
    packed, scale = session.run(None, {"x": x})
    ref = PolarQuantReference(c.key, precomputed_norm=True)
    assert compare_encode(ref, x, packed, scale, FLOAT32_GRAPH)["passed"]


def test_gaussian_objective_fixed_point_and_bounded_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    optimizer = importlib.import_module("optimize_lloyd_max_k8")
    positive = codebook()[128:] * np.sqrt(128)
    mass, first, _ = optimizer.moments(positive)
    assert np.max(np.abs(positive - first / mass)) <= optimizer.CENTROID_TOL
    _, history, converged = optimizer.iterate(
        optimizer.starts()["gaussian_quantiles"], maximum=40
    )
    assert not converged and len(history) == 41
    assert all(b["mse"] <= a["mse"] + 2e-16 for a, b in pairwise(history))
    assert optimizer.mse(positive) < optimizer.mse(optimizer.starts()["legacy_100"])


def test_virtualenv_interpreter_not_dereferenced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    script = importlib.import_module("validate_native_decoder")
    interpreter = tmp_path / "qnn-venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "build",
            "--work-dir",
            str(tmp_path / "probe"),
            "--package",
            str(tmp_path / "native"),
            "--qnn-python",
            str(interpreter),
        ],
    )
    seen = []
    monkeypatch.setattr(script, "build", lambda args: seen.append(args.qnn_python))
    script.main()
    assert seen == [interpreter]
    assert seen[0] != interpreter.resolve()
