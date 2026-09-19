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
    # Cache and graph I/O keep the 16-bit integer grid of the K/V producers
    # (no conversion, no codec). Uncompressed comparison group.
    INT16 = "int16"
    POLAR = "polar"


class Rotation(Enum):
    FWHT = "fwht"
    # Haar QR rotation used by the reference PolarQuant class and default export.
    DENSE_QR = "dense_qr"


@dataclass(frozen=True)
class KVCodecSpec:
    """How one of K or V is stored.

    INT16 and POLAR read the value before the KV-specific int8 encodings of
    an exported w4a16 part (see ``graph_surgery``); BASELINE keeps that path.
    """

    kind: CodecKind
    bits: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind != CodecKind.POLAR and (self.bits != 0 or self.seed != 0):
            raise ValueError(f"{self.kind.name} codec takes no bits or seed.")
        if self.kind == CodecKind.POLAR and self.bits not in (3, 4):
            raise ValueError(
                f"PolarQuant bit width must be 3 or 4, got {self.bits}. "
                "Other widths have no frozen codebook."
            )

    @property
    def is_polar(self) -> bool:
        return self.kind == CodecKind.POLAR

    @property
    def is_int16(self) -> bool:
        return self.kind == CodecKind.INT16

    @property
    def modifies_graph(self) -> bool:
        return self.kind != CodecKind.BASELINE

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "bits": self.bits, "seed": self.seed}


BASELINE = KVCodecSpec(CodecKind.BASELINE)
INT16 = KVCodecSpec(CodecKind.INT16)


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

    def __post_init__(self) -> None:
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
    def codecs(self) -> tuple[KVCodecSpec, KVCodecSpec]:
        return (self.key, self.value)

    @property
    def enabled(self) -> bool:
        """True when at least one KV tensor is stored with a PolarQuant codec."""
        return self.key.is_polar or self.value.is_polar

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
                if self.rotation == Rotation.FWHT:
                    # Keep historical FWHT hashes/cache identities unchanged.
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
# 3-bit profiles have a host oracle only (no HTP graph).
PROFILES: dict[str, TurboQuantConfig] = {
    # Baselines do not rotate; retain their historical metadata/hash.
    "baseline_int8": TurboQuantConfig(
        "baseline_int8", BASELINE, BASELINE, rotation=Rotation.FWHT
    ),
    "baseline_int16_kv": TurboQuantConfig(
        "baseline_int16_kv", INT16, INT16, rotation=Rotation.FWHT
    ),
    "k4_v4": TurboQuantConfig("k4_v4", _polar(4, KEY_SEED), _polar(4, VALUE_SEED)),
    "k4_v4_scaled": TurboQuantConfig(
        "k4_v4_scaled",
        _polar(4, KEY_SEED),
        _polar(4, VALUE_SEED),
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


def get_profile(name: str, rotation: Rotation | None = None) -> TurboQuantConfig:
    """Get a profile; an explicit rotation reproduces historical FWHT bundles."""
    if name == "qjl_reference":
        raise NotImplementedError(
            "qjl_reference is a research option outside the first milestone; "
            "the default path is QJL-off PolarQuant."
        )
    if name not in PROFILES:
        raise ValueError(
            f"Unknown TurboQuant profile '{name}'. Choose from {sorted(PROFILES)}."
        )
    config = PROFILES[name]
    return replace(config, rotation=rotation) if rotation is not None else config
