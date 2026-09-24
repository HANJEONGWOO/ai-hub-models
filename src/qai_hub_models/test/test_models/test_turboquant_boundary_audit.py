# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Compiled 4B key scaling is allowed only at the attention-side boundary."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "early",
        "narrow_concat",
        "multiply",
        "dynamic",
        "vector",
        "wide",
        "unused",
        "non_matmul",
        "wrong_operand",
        "narrow_query",
        "reversed",
    ],
)
def test_scalar_key_division_boundary(
    invalid: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    audit = importlib.import_module("verify_kv_boundary")
    t = audit.Tensor
    concat = audit.Op(
        "0", "concat", "Concat", outputs=[t("cat", "uFxp_16", "1,1,128,1024", "NATIVE")]
    )
    scale = audit.Op(
        "1",
        "key_div",
        "Eltwise_Binary",
        inputs=[
            t("converted", "uFxp_8", "1,1,128,1024", "NATIVE"),
            t("divisor", "uFxp_8", "1", "STATIC"),
        ],
        outputs=[t("scaled", "uFxp_8", "1,1,128,1024", "NATIVE")],
        params=["operation: 2"],
    )
    matmul = audit.Op(
        "2",
        "qk",
        "MatMul",
        inputs=[t("q", "uFxp_16", "1,1,1,128", "NATIVE"), scale.outputs[0]],
    )
    info = audit.DlcInfo(
        [concat, scale, matmul], {"cat": concat}, {"scaled": [matmul]}, {}
    )
    if invalid == "early":
        concat.op_type = "StridedSlice"
    elif invalid == "narrow_concat":
        concat.outputs[0].dtype = "uFxp_8"
    elif invalid == "multiply":
        scale.params = ["operation: 3"]
    elif invalid == "dynamic":
        scale.inputs[1].ttype = "NATIVE"
    elif invalid == "vector":
        scale.inputs[1].dims = "128"
    elif invalid == "wide":
        scale.outputs[0].dtype = "uFxp_16"
    elif invalid == "unused":
        info.consumers.clear()
    elif invalid == "non_matmul":
        matmul.op_type = "Add"
    elif invalid == "wrong_operand":
        matmul.inputs.reverse()
    elif invalid == "narrow_query":
        matmul.inputs[0].dtype = "uFxp_8"
    elif invalid == "reversed":
        scale.inputs.reverse()
    assert audit.is_key_scale_boundary(info, "cat", {"converted"}, scale) == (
        invalid is None
    )
