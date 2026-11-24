# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Profiling attribution must not confuse inclusive counters with stage cost."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"


@pytest.fixture
def profiler(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("summarize_stage_profiles")


@pytest.mark.parametrize(
    ("name", "stage", "kind"),
    [
        ("tq_key_0_enc_unit", "encode.normalize", "key"),
        ("tq_key_0_enc_scale", "encode.normalize", "key"),
        ("tq_key_0_enc_rotated", "encode.rotate", "key"),
        ("tq_value_4_enc_above_i32", "encode.scalar_index", "value"),
        ("tq_key_0_enc_scalar_level0_above", "encode.scalar_index", "key"),
        ("tq_value_4_enc_scalar_level3_select2_0_hi", "encode.scalar_index", "value"),
        ("tq_key_0_enc_scalar_sum3", "encode.scalar_index", "key"),
        ("tq_value_4_enc_scale_len", "encode.scale_correction", "value"),
        ("tq_value_4_enc_index_hi", "encode.pack", "value"),
        ("tq_key_0_enc_pack_columns", "encode.pack", "key"),
        ("tq_key_0_enc_pack_byte0", "encode.pack", "key"),
        ("tq_value_4_enc_pack_pairs", "encode.pack", "value"),
        ("tq_key_27_tile768_native_fp16", "decode.native_past", "key"),
        ("tq_value_27_current_native_fp16", "decode.native_current", "value"),
        ("tq_attn_3_head0_q1_rotated", "attention.query_rotate", "query"),
        ("tq_attn_3_tile512_head0_q1_score", "attention.qk", "key"),
        ("tq_attn_3_tile512_head0_q1_sum", "attention.av", "value"),
        ("fp16_attn_3_head0_q1_av_half", "attention.av", "value"),
    ],
)
def test_semantic_stages(
    profiler: ModuleType, name: str, stage: str, kind: str
) -> None:
    actual, layer, kv = profiler.semantic_stage(name)
    assert actual == stage
    assert layer is not None
    assert kv == kind


def test_only_leaf_node_cycles(profiler: ModuleType) -> None:
    events = [
        {"type": 3003, "unit": 3, "children": 2, "value": 100},
        {"type": 404, "unit": 3, "children": 1, "value": 90},
        {"type": 404, "unit": 3, "children": 0, "value": 80},
        {"type": 404, "unit": 3, "children": 0, "value": 0},
        {"type": 404, "unit": 1, "children": 0, "value": 20},
    ]
    assert profiler.leaf_nodes(events) == events[2:4]


def test_zero_events_not_claimed_as_no_work(profiler: ModuleType) -> None:
    result = profiler.aggregate(
        [
            {"stage": "encode.rotate", "cycles": 0, "kv": "key"},
            {"stage": "decode.native_past", "cycles": 100, "kv": "key"},
        ]
    )
    assert result[0]["percent_leaf_cycles"] == 100
    assert result[1]["zero_events"] == 1
    assert result[1]["mean_cycles_per_nonzero_event"] is None


def test_rotation_layout_is_not_gemm(profiler: ModuleType) -> None:
    from verify_kv_boundary import Op

    graph = profiler.GraphStages.__new__(profiler.GraphStages)
    graph.source = {}
    name = "tq_key_0_enc_rotated_ht1d"
    graph.compiled = {name: Op("1", name, "Transpose")}
    assert graph.classify(name + ":OpId_22 (cycles)")["stage"] == "layout_precision"


def test_converter_cast_not_query_rotation(profiler: ModuleType) -> None:
    graph = profiler.GraphStages.__new__(profiler.GraphStages)
    graph.source = {}
    graph.compiled = {}
    name = "tq_attn_0_head0_q0_rotated_pre_reshape_converted_QNN_DATATYPE_FLOAT_16"
    assert graph.classify(name + ":OpId_123 (cycles)")["stage"] == "layout_precision"


@pytest.fixture
def optrace(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("summarize_optrace")


def test_optrace_interval_union_not_parallel_sum(optrace: ModuleType) -> None:
    assert optrace.union_length([(10, 30), (20, 40), (45, 50)]) == 35
    assert optrace.union_length([]) == 0


def test_optrace_recovers_hmx_rotation_but_not_layout(
    optrace: ModuleType, profiler: ModuleType
) -> None:
    from verify_kv_boundary import Op

    graph = profiler.GraphStages.__new__(profiler.GraphStages)
    graph.source = {}
    graph.compiled = {
        "tq_key_0_enc_rotated": Op("1", "tq_key_0_enc_rotated", "FullyConnected"),
        "tq_key_0_enc_rotated_post_reshape": Op(
            "2", "tq_key_0_enc_rotated_post_reshape", "Reshape"
        ),
    }
    event = {
        "name": "q::ConvLayer.fp16.s1.tcm",
        "args": {
            "QNN Op Name": "tq_key_0_enc_rotated_post_reshape",
            "Flags": ["uses_hmx"],
        },
    }
    assert optrace.classify_kernel(graph, event)["stage"] == "encode.rotate"
    event["name"] = "q::ForceFormat_Crouton"
    event["args"]["Flags"] = ["uses_hvx"]
    assert optrace.classify_kernel(graph, event)["stage"] == "layout_precision"


def test_optrace_no_arithmetic_parent_no_reclassification(
    optrace: ModuleType, profiler: ModuleType
) -> None:
    graph = profiler.GraphStages.__new__(profiler.GraphStages)
    graph.source = {}
    graph.compiled = {}
    event = {
        "name": "q::ConvLayer.fp16.s1.tcm",
        "args": {
            "QNN Op Name": "tq_key_0_enc_rotated_post_reshape",
            "Flags": ["uses_hmx"],
        },
    }
    assert optrace.classify_kernel(graph, event)["stage"] == "layout_precision"


def test_optrace_excludes_duplicate_views_and_nonexecuted(optrace: ModuleType) -> None:
    event = {
        "ph": "X",
        "pid": 0,
        "tid": 512,
        "ts": 10,
        "dur": 12,
        "args": {"ID": "abc", "Duration (cycles)": 11, "Start Cycle": 100},
    }
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "pid": 0,
                "name": "process_name",
                "args": {"name": "Core 0 Overview"},
            },
            {
                "ph": "M",
                "pid": 0,
                "tid": 512,
                "name": "thread_name",
                "args": {"name": "Type: HVX"},
            },
            {
                "ph": "M",
                "pid": 0,
                "tid": 1,
                "name": "thread_name",
                "args": {"name": "Non Executed Tensors"},
            },
            event,
            {**event, "pid": 3},
            {**event, "tid": 1},
        ]
    }
    physical = optrace.physical_events(trace)
    assert len(physical) == 1
    # Shared per-op statistics can differ from the actual per-worker interval.
    assert physical[0]["dur"] == 12
    trace["traceEvents"].append(event)
    with pytest.raises(ValueError, match="Duplicate physical event"):
        optrace.physical_events(trace)


def test_optrace_requires_explicit_physical_track(optrace: ModuleType) -> None:
    with pytest.raises(ValueError, match="Missing Core Overview"):
        optrace.physical_events({"traceEvents": []})
