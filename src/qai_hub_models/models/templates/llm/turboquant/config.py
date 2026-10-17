# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Opt-in TurboQuant KV-cache codec configuration.

A config is immutable and hashable, and ``config_hash()`` covers every field
plus the digests of the codebooks and rotation constants it selects, so it can
key model instance caches and artifact directories.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from qai_hub_models.models.templates.llm.turboquant.constants import (
    CODEBOOK_HEX,
    CODEBOOK_SHA256,
    FWHT_SIGNS,
    FWHT_SIGNS_SHA256,
    REFERENCE_COMMIT,
)

FORMAT_NAME = "qaihm-turboquant-kv"
FORMAT_VERSION = 1

# Reference KVCacheCompressor seed policy (K = seed, V = seed + 500).
KEY_SEED = 42
VALUE_SEED = 542


class CodecKind(Enum):
    # The repo's existing KV path (int8 affine on device). The codec leaves it untouched.
    BASELINE = "baseline"
    # Uncompressed cache with a shared 16-bit affine grid per layer/KV kind.
    # Head-grid conversion and the legacy int8 attention boundary can remain.
    INT16 = "int16"
    # Uncompressed FP16 cache, including current KV, and FP16 QK/AV inputs.
    FP16 = "fp16"
    POLAR = "polar"


class Rotation(Enum):
    FWHT = "fwht"
    # Haar QR rotation used by the reference PolarQuant class and default export.
    DENSE_QR = "dense_qr"


@dataclass(frozen=True)
class KVCodecSpec:
    """How one of K or V is stored.

    INT16, FP16 and POLAR read before the KV-specific int8 encodings of
    an exported w4a16 part (see ``graph_surgery``); BASELINE keeps that path.
    """

    kind: CodecKind
    bits: int = 0
    seed: int = 0
    quantizer: str = "lloyd_max"

    def __post_init__(self) -> None:
        if self.quantizer not in ("lloyd_max", "structured_k8_two_region_v1"):
            raise ValueError("Unknown scalar quantizer")
        if self.quantizer != "lloyd_max" and (
            self.kind != CodecKind.POLAR or self.bits != 8
        ):
            raise ValueError("Structured quantizer requires Polar K8")
        if self.kind != CodecKind.POLAR and (self.bits != 0 or self.seed != 0):
            raise ValueError(f"{self.kind.name} codec takes no bits or seed.")
        if self.kind == CodecKind.POLAR and self.bits not in range(2, 9):
            raise ValueError(
                f"PolarQuant bit width must be 2 through 8, got {self.bits}. "
                "Other widths have no frozen codebook."
            )

    @property
    def is_polar(self) -> bool:
        return self.kind == CodecKind.POLAR

    @property
    def is_int16(self) -> bool:
        return self.kind == CodecKind.INT16

    @property
    def is_fp16(self) -> bool:
        return self.kind == CodecKind.FP16

    @property
    def modifies_graph(self) -> bool:
        return self.kind != CodecKind.BASELINE

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": self.kind.value, "bits": self.bits, "seed": self.seed}
        if self.quantizer != "lloyd_max":
            result["quantizer"] = self.quantizer
        return result


BASELINE = KVCodecSpec(CodecKind.BASELINE)
INT16 = KVCodecSpec(CodecKind.INT16)
FP16 = KVCodecSpec(CodecKind.FP16)


@dataclass(frozen=True)
class TurboQuantConfig:
    """PolarQuant KV codec settings with opt-in K3+1 orthogonal QJL."""

    profile: str
    key: KVCodecSpec
    value: KVCodecSpec
    block_size: int = 128
    rotation: Rotation = Rotation.DENSE_QR
    norm_correction: bool = True
    norm_dtype: str = "float16"
    bit_order: str = "msb_first"
    format_version: int = FORMAT_VERSION
    precomputed_norm: bool = False
    qjl: bool = False
    reference_commit: str = field(default=REFERENCE_COMMIT)
    # Empty retains every legacy profile's serialization/hash. Layer indices
    # are absolute model layer IDs, not part-local IDs.
    layer_bits: tuple[tuple[int, int], ...] = ()
    layer_policy: str = ""
    structured_k8: bool = False

    def codec_for(self, kind: str, layer: int) -> KVCodecSpec:
        if kind not in ("key", "value"):
            raise ValueError(f"Unknown KV kind: {kind}")
        spec = getattr(self, kind)
        if not self.layer_bits:
            return spec
        if not 0 <= layer < len(self.layer_bits):
            raise ValueError(f"Layer {layer} is outside the mixed-bit plan.")
        bits = self.layer_bits[layer][kind == "value"]
        return replace(
            spec,
            bits=bits,
            quantizer=(
                "structured_k8_two_region_v1"
                if self.structured_k8 and kind == "key" and bits == 8
                else "lloyd_max"
            ),
        )

    def for_layer(self, layer: int) -> TurboQuantConfig:
        if not self.layer_bits:
            return self
        return replace(
            self,
            key=self.codec_for("key", layer),
            value=self.codec_for("value", layer),
            layer_bits=(),
            layer_policy="",
        )

    def __post_init__(self) -> None:
        if self.structured_k8 and (
            self.block_size != 128
            or self.qjl
            or not self.precomputed_norm
            or not self.norm_correction
            or self.norm_dtype != "float16"
            or self.rotation != Rotation.DENSE_QR
        ):
            raise ValueError(
                "Structured K8 requires Dense D128, norm-corrected FP16 scales, no QJL"
            )
        if self.value.quantizer != "lloyd_max" or (
            self.key.quantizer != "lloyd_max" and not self.structured_k8
        ):
            raise ValueError("Structured quantizer is opt-in K-only")
        if self.layer_bits:
            if not isinstance(self.layer_bits, tuple) or any(
                not isinstance(pair, tuple) or any(type(bit) is not int for bit in pair)
                for pair in self.layer_bits
            ):
                raise ValueError(
                    "Layer plan must be an immutable tuple of integer bit pairs."
                )
            if (
                self.qjl
                or not self.precomputed_norm
                or not all(s.is_polar for s in (self.key, self.value))
            ):
                raise ValueError(
                    "Mixed-bit requires scaled PolarQuant K/V without QJL."
                )
            if self.layer_policy not in ("uniform", "attention_mse", "nll_allocation"):
                raise ValueError(
                    "Mixed-bit requires a recorded layer selection policy."
                )
            if any(len(pair) != 2 for pair in self.layer_bits):
                raise ValueError("Each layer requires exactly K and V bit widths.")
        elif self.layer_policy:
            raise ValueError("A layer policy requires a layer plan.")
        if any(s.is_fp16 for s in self.codecs) and not all(
            s.is_fp16 for s in self.codecs
        ):
            raise ValueError(
                "FP16 attention requires both K and V to use FP16 storage."
            )
        if self.norm_dtype not in ("float16", "float32"):
            raise ValueError(f"Unsupported norm dtype {self.norm_dtype}.")
        if self.bit_order != "msb_first":
            raise ValueError(f"Unsupported bit order {self.bit_order}.")
        expected_version = (
            3 if self.qjl else 2 if self.precomputed_norm else FORMAT_VERSION
        )
        if self.format_version != expected_version:
            raise ValueError(
                f"Config format version {self.format_version} does not match "
                f"this norm representation ({expected_version})."
            )
        if self.qjl and (
            not self.precomputed_norm
            or not self.norm_correction
            or self.key.bits != 3
            or self.value.bits != 4
            or self.rotation != Rotation.DENSE_QR
            or self.norm_dtype != "float16"
        ):
            raise ValueError("QJL requires dense K3+1/V4 with precomputed FP16 scales.")
        for spec in self.codecs:
            if not spec.is_polar:
                continue
            if (spec.bits, self.block_size) not in CODEBOOK_HEX:
                raise ValueError(
                    f"No frozen {spec.bits}-bit codebook for block size "
                    f"{self.block_size}; regenerate constants.py first."
                )
            if (
                self.rotation == Rotation.FWHT
                and (spec.seed, self.block_size) not in FWHT_SIGNS
            ):
                raise ValueError(
                    f"No frozen FWHT signs for seed {spec.seed}, block size "
                    f"{self.block_size}; regenerate constants.py first."
                )

    @property
    def codecs(self) -> tuple[KVCodecSpec, ...]:
        if not self.layer_bits:
            return (self.key, self.value)
        return tuple(
            self.codec_for(kind, layer)
            for layer in range(len(self.layer_bits))
            for kind in ("key", "value")
        )

    @property
    def enabled(self) -> bool:
        """True when at least one KV tensor is stored with a PolarQuant codec."""
        return self.key.is_polar or self.value.is_polar

    @property
    def fp16_attention(self) -> bool:
        return self.key.is_fp16 and self.value.is_fp16

    @property
    def modifies_graph(self) -> bool:
        """True when the exported part needs surgery (codec or float16 KV)."""
        return self.key.modifies_graph or self.value.modifies_graph

    def validate_for_model(
        self, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> None:
        """Reject model shapes this format version cannot store without truncation."""
        if num_layers <= 0 or num_kv_heads <= 0:
            raise ValueError("num_layers and num_kv_heads must be positive.")
        if self.layer_bits and len(self.layer_bits) != num_layers:
            raise ValueError(
                "Mixed-bit layer plan length differs from model layer count."
            )
        if not self.enabled:
            return
        if head_dim & (head_dim - 1) or head_dim <= 0:
            raise ValueError(
                f"head_dim={head_dim} is not a power of two; FWHT padding is not "
                f"supported in format version {self.format_version}."
            )
        if head_dim != self.block_size:
            raise ValueError(
                f"head_dim={head_dim} must equal block_size={self.block_size} "
                f"in format version {self.format_version} (one scalar per head vector)."
            )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "format": FORMAT_NAME,
            "format_version": self.format_version,
            "profile": self.profile,
            "key": self.key.to_dict(),
            "value": self.value.to_dict(),
            "block_size": self.block_size,
            "rotation": self.rotation.value,
            "norm_correction": self.norm_correction,
            "norm_dtype": self.norm_dtype,
            "bit_order": self.bit_order,
            "qjl": self.qjl,
            "reference_commit": self.reference_commit,
        }
        if self.precomputed_norm:
            data["norm_representation"] = "effective_scale"
        if self.layer_bits:
            data["layer_plan"] = {
                "bits": [list(pair) for pair in self.layer_bits],
                "policy": self.layer_policy,
                "codebook_sha256": {
                    str(bits): CODEBOOK_SHA256[(bits, self.block_size)]
                    for bits in sorted({s.bits for s in self.codecs})
                },
                "average_payload_bits": sum(sum(pair) for pair in self.layer_bits)
                / len(self.layer_bits),
                "packing": "msb_first;8bit=uint8_index",
            }
        if self.enabled:
            # Fixed implementation, not a selectable strategy. Invalidate old
            # broadcast bundles even though the codebook/cache ABI is unchanged.
            data["scalar_indexing"] = "lloyd_tree"
        if self.structured_k8:
            from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
                metadata,
            )

            data["structured_k8"] = metadata()
            data["scalar_indexing"] = "lloyd_tree_except_structured_k8_arithmetic"
            if self.layer_bits:
                data["layer_plan"]["codebook_sha256"]["8"] = metadata()[
                    "codebook_sha256"
                ]
        if self.fp16_attention:
            data["attention"] = {
                "kv_storage": "float16",
                "qk_av_inputs": "float16",
                "current_kv": "read_stored_fp16",
                "rotation": "none",
                "recalibrated": False,
            }
        if self.qjl:
            from qai_hub_models.models.templates.llm.turboquant.qjl import projection

            data["qjl_settings"] = {
                "reference_commit": "7f601a135735842a7f12b6bf861561154c410ff4",
                "keys_only": True,
                "seed": self.key.seed + 1000,
                "projection": "orthogonal_qr",
                "coefficient": "sqrt(pi/2)/sqrt(d)",
                "shrinkage": 1.0,
                "packing": "nibble_low3_mse_high1_positive_sign",
                "residual": "original_key_minus_native_fp16_mse_reconstruction",
                "scale": "coefficient_times_residual_norm_fp16",
                "projection_f32_sha256": hashlib.sha256(
                    projection(self.block_size, self.key.seed + 1000)
                    .astype("<f4")
                    .tobytes()
                ).hexdigest(),
            }
        for name, spec in (("key", self.key), ("value", self.value)):
            if spec.is_polar:
                data[name]["codebook_sha256"] = CODEBOOK_SHA256[
                    (spec.bits, self.block_size)
                ]
                if spec.quantizer != "lloyd_max":
                    from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
                        CODEBOOK_SHA256 as structured_hash,
                    )

                    data[name]["codebook_sha256"] = structured_hash
                if self.rotation == Rotation.FWHT:
                    # Preserve the FWHT sign hashes; indexing identity is separate.
                    data[name]["signs_sha256"] = FWHT_SIGNS_SHA256[
                        (spec.seed, self.block_size)
                    ]
                else:
                    # Hash the actual exported R, not just its seed: QR can
                    # depend on the host LAPACK build. R.T is derived from R.
                    from qai_hub_models.models.templates.llm.turboquant.reference import (
                        make_rotation,
                    )

                    matrix = make_rotation(
                        self.rotation, spec.seed, self.block_size
                    ).matrix()
                    data[name]["rotation_f32_sha256"] = hashlib.sha256(
                        matrix.astype("<f4").tobytes()
                    ).hexdigest()
        return data

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _polar(bits: int, seed: int) -> KVCodecSpec:
    return KVCodecSpec(CodecKind.POLAR, bits=bits, seed=seed)


# "k8" names the repo's affine int8 K path, not an 8-bit TurboQuant codec.
# Unscaled legacy 3-bit profiles have a host oracle only (no HTP graph).
PROFILES: dict[str, TurboQuantConfig] = {
    # Baselines do not rotate; retain their historical metadata/hash.
    "baseline_int8": TurboQuantConfig(
        "baseline_int8", BASELINE, BASELINE, rotation=Rotation.FWHT
    ),
    "baseline_int16_kv": TurboQuantConfig(
        "baseline_int16_kv", INT16, INT16, rotation=Rotation.FWHT
    ),
    "baseline_fp16_kv_fp16_attn": TurboQuantConfig(
        "baseline_fp16_kv_fp16_attn", FP16, FP16
    ),
    "k4_v4": TurboQuantConfig("k4_v4", _polar(4, KEY_SEED), _polar(4, VALUE_SEED)),
    "k4_v4_scaled": TurboQuantConfig(
        "k4_v4_scaled",
        _polar(4, KEY_SEED),
        _polar(4, VALUE_SEED),
        format_version=2,
        precomputed_norm=True,
    ),
    "k5_v3_scaled": TurboQuantConfig(
        "k5_v3_scaled",
        _polar(5, KEY_SEED),
        _polar(3, VALUE_SEED),
        format_version=2,
        precomputed_norm=True,
    ),
    "k6_v2_scaled": TurboQuantConfig(
        "k6_v2_scaled",
        _polar(6, KEY_SEED),
        _polar(2, VALUE_SEED),
        format_version=2,
        precomputed_norm=True,
    ),
    "k3qjl_v4_scaled": TurboQuantConfig(
        "k3qjl_v4_scaled",
        _polar(3, KEY_SEED),
        _polar(4, VALUE_SEED),
        format_version=3,
        precomputed_norm=True,
        qjl=True,
    ),
    "k8_v3": TurboQuantConfig("k8_v3", BASELINE, _polar(3, VALUE_SEED)),
    "k4_v3": TurboQuantConfig("k4_v3", _polar(4, KEY_SEED), _polar(3, VALUE_SEED)),
}


for _budget, _period in ((5, 4), (6, 2)):
    _name = f"mixed_k{_budget}_v{8 - _budget}_uniform"
    PROFILES[_name] = TurboQuantConfig(
        _name,
        _polar(4, KEY_SEED),
        _polar(2, VALUE_SEED),
        format_version=2,
        precomputed_norm=True,
        layer_policy="uniform",
        layer_bits=tuple(
            (
                8 if (layer + 1) % _period == 0 else 4,
                4 if _budget == 5 and layer % 2 else 2,
            )
            for layer in range(28)
        ),
    )
PROFILES["k8_v4_scaled"] = TurboQuantConfig(
    "k8_v4_scaled",
    _polar(8, KEY_SEED),
    _polar(4, VALUE_SEED),
    format_version=2,
    precomputed_norm=True,
)
PROFILES["mixed_k6_v2_uniform_structured_k8"] = replace(
    PROFILES["mixed_k6_v2_uniform"],
    profile="mixed_k6_v2_uniform_structured_k8",
    structured_k8=True,
)
PROFILES["structured_k8_v4_scaled"] = replace(
    PROFILES["k8_v4_scaled"],
    profile="structured_k8_v4_scaled",
    structured_k8=True,
    key=KVCodecSpec(CodecKind.POLAR, 8, KEY_SEED, "structured_k8_two_region_v1"),
)

STRUCTURED_ALLOCATION_PROFILE = "mixed_k6_v2_structured_k8_allocation"
_STRUCTURED_ALLOCATION_BASE = replace(
    PROFILES["mixed_k6_v2_uniform_structured_k8"],
    profile=STRUCTURED_ALLOCATION_PROFILE,
    layer_policy="nll_allocation",
)


def structured_allocation_plan(k8_layers: object) -> dict[str, Any]:
    """Create a canonical, hash-checked K8x14/K4x14/V2 allocation artifact.

    The policy describes this opt-in experiment, not its selection objective;
    fixed masks and NLL-selected masks use the same inference identity scheme.
    Diagnostic single-layer flips are deliberately not deployable here.
    """
    if (
        not isinstance(k8_layers, list)
        or len(k8_layers) != 14
        or any(type(layer) is not int or not 0 <= layer < 28 for layer in k8_layers)
        or len(set(k8_layers)) != 14
    ):
        raise ValueError(
            "Allocation requires exactly 14 distinct integer layers in [0,27]"
        )
    layers = sorted(k8_layers)
    config = replace(
        _STRUCTURED_ALLOCATION_BASE,
        layer_bits=tuple((8 if layer in layers else 4, 2) for layer in range(28)),
    )
    return {
        "profile": STRUCTURED_ALLOCATION_PROFILE,
        "policy": "nll_allocation",
        "k8_layers": layers,
        "bits": [list(pair) for pair in config.layer_bits],
        "config_hash": config.config_hash(),
        "config": config.to_dict(),
    }


def get_profile(
    name: str,
    rotation: Rotation | None = None,
    *,
    layer_plan: dict[str, Any] | None = None,
) -> TurboQuantConfig:
    """Get a profile; an explicit rotation selects the legacy FWHT rotation."""
    if name == "qjl_reference":
        raise NotImplementedError(
            "qjl_reference is a research option outside the first milestone; "
            "the default path is QJL-off PolarQuant."
        )
    lookup = name.replace("_mse", "_uniform") if name.startswith("mixed_") else name
    if lookup not in PROFILES and name != STRUCTURED_ALLOCATION_PROFILE:
        raise ValueError(
            f"Unknown TurboQuant profile '{name}'. Choose from {sorted(PROFILES)}."
        )
    config = (
        _STRUCTURED_ALLOCATION_BASE
        if name == STRUCTURED_ALLOCATION_PROFILE
        else PROFILES[lookup]
    )
    if name == STRUCTURED_ALLOCATION_PROFILE:
        if rotation is not None and rotation != Rotation.DENSE_QR:
            raise ValueError("Structured allocation requires Dense QR")
        if layer_plan is None:
            raise ValueError(
                "Structured allocation requires its frozen layer-plan JSON"
            )
        expected = structured_allocation_plan(layer_plan.get("k8_layers"))
        if any(layer_plan.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Allocation plan/config/codebook hash or bit budget mismatch"
            )
        config = replace(
            config, layer_bits=tuple(tuple(pair) for pair in expected["bits"])
        )
    elif name.endswith("_mse"):
        if rotation is not None and rotation != config.rotation:
            raise ValueError(
                "MSE-selected layer plans are fixed to their Dense rotation."
            )
        if (
            layer_plan is None
            or layer_plan.get("profile") != name
            or layer_plan.get("policy") != "attention_mse"
        ):
            raise ValueError(
                "MSE profile requires its frozen, matching layer-plan JSON."
            )
        pairs = tuple(tuple(pair) for pair in layer_plan["bits"])
        if sorted(p[0] for p in pairs) != sorted(
            p[0] for p in config.layer_bits
        ) or sorted(p[1] for p in pairs) != sorted(p[1] for p in config.layer_bits):
            raise ValueError("Layer plan violates the exact mixed-bit budget/quotas.")
        config = replace(
            config, profile=name, layer_bits=pairs, layer_policy="attention_mse"
        )
        if layer_plan.get("config_hash") != config.config_hash():
            raise ValueError("Layer plan/config/codebook hash mismatch.")
    elif layer_plan is not None:
        raise ValueError(
            "External layer plans are only accepted for mixed MSE profiles."
        )
    return replace(config, rotation=rotation) if rotation is not None else config
