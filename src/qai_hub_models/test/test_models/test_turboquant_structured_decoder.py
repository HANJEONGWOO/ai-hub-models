# SPDX-License-Identifier: BSD-3-Clause
"""Arithmetic decoder changes only K8 reconstruction, not bytes or attention."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    quantize_current_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    STRUCTURED_COEFFICIENTS,
    STRUCTURED_OP,
    use_native_decoder,
    use_structured_arithmetic_decoder,
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
    codebook,
    coefficients,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models import test_turboquant_tiled_attention as fixture


@pytest.mark.parametrize("tokens", [1, 3, 256, 1023])
def test_arithmetic_reference_bit_exact(tokens: int) -> None:
    packed = np.resize(np.arange(256, dtype=np.uint8), (8, 1, tokens, 128))
    edges = np.array(
        [
            0,
            0x8000,
            1,
            0x3FF,
            0x400,
            0x401,
            0x3BFF,
            0x3C00,
            0x3C01,
            0x4000,
            0x7BFF,
            0x8001,
            0x8400,
            0xBC00,
            0xFBFF,
        ],
        np.uint16,
    ).view(np.float16)
    scale = np.resize(np.repeat(edges, 2), (8, 1, tokens, 1))
    outputs = []
    for op, values in [("Decode8", codebook()), (STRUCTURED_OP, coefficients())]:
        graph = helper.make_graph(
            [
                helper.make_node(
                    op, ["packed", "scale", "params"], ["decoded"], domain="turboquant"
                )
            ],
            "test",
            [
                helper.make_tensor_value_info(
                    "packed", TensorProto.UINT8, list(packed.shape)
                ),
                helper.make_tensor_value_info(
                    "scale", TensorProto.FLOAT16, list(scale.shape)
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "decoded", TensorProto.FLOAT16, list(packed.shape)
                )
            ],
            [
                numpy_helper.from_array(
                    values.astype(np.float16).reshape(1, 1, 1, -1), "params"
                )
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 17),
                helper.make_opsetid("turboquant", 1),
            ],
            ir_version=9,
        )
        model = with_reference_decoder(model)
        onnx.checker.check_model(model, full_check=True)
        # ORT CPU FP16 promotion can elide the intermediate FP16 rounding even
        # with graph optimizations disabled. Use ONNX's literal evaluator for
        # the rounding contract; the actual HVX kernel is independently tested.
        outputs.append(
            ReferenceEvaluator(model).run(None, {"packed": packed, "scale": scale})[0]
        )
    expected = (
        codebook().astype(np.float16)[packed].astype(np.float32)
        * scale.astype(np.float32)
    ).astype(np.float16)
    for output in outputs:
        np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("seq", [1, 3, 128])
@pytest.mark.parametrize("current", [False, True])
def test_decoder_only_surgery(
    seq: int, current: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = max(35, seq + 17)
    monkeypatch.setattr(fixture, "CONTEXT", context)
    config = get_profile("structured_k8_v4_scaled")
    model, enc = fixture.attention_part(seq)
    before = use_native_decoder(
        tile_kv_attention(
            apply_kv_profile(model, enc, config, seq, context), config, 7, rotated=True
        ),
        config,
    )
    if current:
        before = quantize_current_attention(before, config)
    saved = before.model.SerializeToString()
    after = use_structured_arithmetic_decoder(before, config)
    assert before.model.SerializeToString() == saved
    assert before.encodings == after.encodings
    assert config.config_hash() == get_profile("structured_k8_v4_scaled").config_hash()
    for side in ("input", "output", "value_info"):
        assert [x.SerializeToString() for x in getattr(before.model.graph, side)] == [
            x.SerializeToString() for x in getattr(after.model.graph, side)
        ]
    expected_count = sum(n.op_type == "Decode8" for n in before.model.graph.node)
    assert expected_count == sum(
        n.op_type == STRUCTURED_OP for n in after.model.graph.node
    )
    for left, right in zip(
        before.model.graph.node, after.model.graph.node, strict=True
    ):
        if right.op_type == STRUCTURED_OP:
            assert right.input[:2] == left.input[:2] and right.output == left.output
            assert right.input[2] == STRUCTURED_COEFFICIENTS
        else:
            assert left.SerializeToString() == right.SerializeToString()
    onnx.checker.check_model(with_reference_decoder(after.model), full_check=True)
    with pytest.raises(ValueError, match="Missing/mismatched Structured K8 LUT"):
        use_structured_arithmetic_decoder(after, config)


def test_wrong_quantizer_rejected_and_k4_layer_unchanged() -> None:
    model, enc = fixture.attention_part(1)
    for profile in ("k8_v4_scaled", "mixed_k6_v2_uniform_structured_k8"):
        config = get_profile(profile)
        graph = use_native_decoder(
            tile_kv_attention(
                apply_kv_profile(model, enc, config, 1, fixture.CONTEXT),
                config,
                7,
                rotated=True,
            ),
            config,
        )
        if profile == "k8_v4_scaled":
            with pytest.raises(ValueError, match="opt-in Structured K8"):
                use_structured_arithmetic_decoder(graph, config)
        else:
            result = use_structured_arithmetic_decoder(graph, config)
            assert result.model.SerializeToString() == graph.model.SerializeToString()
