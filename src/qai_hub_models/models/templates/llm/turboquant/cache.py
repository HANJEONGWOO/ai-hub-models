# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Host-side packed KV cache for the Qwen3 delta-cache ABI (format version 1).

The exported graphs return only the new tokens' KV in hub layout, key
``(kv_heads, batch, head_dim, new)`` and value ``(kv_heads, batch, new, head_dim)``.
This cache encodes just those tokens and keeps them packed in preallocated
``capacity``-sized buffers laid out ``(kv_heads, batch, token, bytes)`` for both
K and V, so the token axis is always -2 regardless of the hub layout.

``BASELINE`` tensors are kept as float32, exactly what the existing host
generator stores; their int8 treatment happens inside the deployed graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from qai_hub_models.models.templates.llm.turboquant.config import (
    FORMAT_NAME,
    FORMAT_VERSION,
    KVCodecSpec,
    TurboQuantConfig,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    packed_nbytes,
    to_storage_norms,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
)

ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(t: ArrayLike) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu", torch.float32).numpy()
    return np.asarray(t)


@dataclass
class CacheMemoryReport:
    """Byte accounting that separates logical use from allocation."""

    tokens: int
    capacity: int
    packed_payload_bytes: int
    norm_bytes: int
    baseline_float_bytes: int
    allocated_bytes: int
    formula_payload_bytes: int

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class PackedKVStore:
    """One of K or V for one layer."""

    def __init__(
        self,
        spec: KVCodecSpec,
        config: TurboQuantConfig,
        num_kv_heads: int,
        head_dim: int,
        capacity: int,
        batch_size: int,
    ) -> None:
        self.spec = spec
        self.head_dim = head_dim
        self.capacity = capacity
        lead = (num_kv_heads, batch_size, capacity)
        self.codec: PolarQuantReference | None = None
        self.packed: np.ndarray | None = None
        self.norms: np.ndarray | None = None
        self.raw: np.ndarray | None = None
        if spec.is_polar:
            self.codec = PolarQuantReference(
                spec, config.block_size, config.rotation, config.norm_correction
            )
            self.packed = np.zeros(
                (*lead, packed_nbytes(head_dim, spec.bits)), np.uint8
            )
            self.norms = np.zeros((*lead, 1), config.norm_dtype)
        else:
            self.raw = np.zeros((*lead, head_dim), np.float32)
        self.norm_dtype = config.norm_dtype

    def write(self, start: int, tokens: np.ndarray) -> None:
        """Encode ``(kv_heads, batch, new, head_dim)`` tokens into ``[start, start + new)``."""
        stop = start + tokens.shape[2]
        if self.codec is None:
            assert self.raw is not None
            self.raw[:, :, start:stop] = tokens
            return
        assert self.packed is not None and self.norms is not None
        indices, norms = self.codec.encode(tokens)
        self.packed[:, :, start:stop] = pack_indices(indices, self.spec.bits)
        self.norms[:, :, start:stop] = to_storage_norms(norms, self.norm_dtype)

    def read(self, length: int) -> np.ndarray:
        """Decode the first ``length`` tokens as float32 ``(kv_heads, batch, length, head_dim)``."""
        if self.codec is None:
            assert self.raw is not None
            return self.raw[:, :, :length].copy()
        assert self.packed is not None and self.norms is not None
        indices = unpack_indices(
            self.packed[:, :, :length], self.spec.bits, self.head_dim
        )
        return self.codec.decode(indices, self.norms[:, :, :length]).astype(np.float32)

    def reset(self) -> None:
        for buf in (self.packed, self.norms, self.raw):
            if buf is not None:
                buf.fill(0)

    def arrays(self) -> dict[str, np.ndarray]:
        found = {"packed": self.packed, "norms": self.norms, "raw": self.raw}
        return {k: v for k, v in found.items() if v is not None}


class TurboQuantKVCache:
    """Packed KV state across prefill chunks and decode steps for one session.

    Only batch size 1 is supported. Appending past ``context_length`` raises
    instead of sliding, and ``reset()`` clears all state for a new session.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        context_length: int,
        batch_size: int = 1,
    ) -> None:
        if batch_size != 1:
            raise NotImplementedError("TurboQuant KV cache supports batch_size=1 only.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        config.validate_for_model(num_layers, num_kv_heads, head_dim)
        self.config = config
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.context_length = context_length
        self.batch_size = batch_size
        self.layers = [
            (
                PackedKVStore(
                    config.key,
                    config,
                    num_kv_heads,
                    head_dim,
                    context_length,
                    batch_size,
                ),
                PackedKVStore(
                    config.value,
                    config,
                    num_kv_heads,
                    head_dim,
                    context_length,
                    batch_size,
                ),
            )
            for _ in range(num_layers)
        ]
        self._length = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._length

    def reset(self) -> None:
        for key_store, value_store in self.layers:
            key_store.reset()
            value_store.reset()
        self._length = 0

    def _check_new_kv(self, key: np.ndarray, value: np.ndarray) -> int:
        heads, batch, dim, new = key.shape
        if (heads, batch, dim) != (self.num_kv_heads, self.batch_size, self.head_dim):
            raise ValueError(f"Key shape {key.shape} does not match the cache layout.")
        if value.shape != (heads, batch, new, dim):
            raise ValueError(
                f"Value shape {value.shape} must be {(heads, batch, new, dim)}; "
                "keys are (kv_heads, batch, head_dim, new) and values "
                "(kv_heads, batch, new, head_dim)."
            )
        return new

    def append(self, flat_kv: Sequence[ArrayLike]) -> None:
        """Encode one step of new-token KV ``[k0, v0, k1, v1, ...]`` in hub layout.

        Shapes cannot tell hub K from HF K when ``new == head_dim``, so callers must
        map tensors by graph I/O name rather than rely on this shape check.
        """
        if len(flat_kv) != 2 * self.num_layers:
            raise ValueError(
                f"Expected {2 * self.num_layers} KV tensors, got {len(flat_kv)}."
            )
        pairs = [
            (_to_numpy(flat_kv[2 * i]), _to_numpy(flat_kv[2 * i + 1]))
            for i in range(self.num_layers)
        ]
        new_counts = {self._check_new_kv(k, v) for k, v in pairs}
        if len(new_counts) != 1:
            raise ValueError(
                f"Layers disagree on new token count: {sorted(new_counts)}."
            )
        new = new_counts.pop()
        if self._length + new > self.context_length:
            raise ValueError(
                f"Context length exhausted: {self._length} cached token(s) plus "
                f"{new} new exceeds context_length={self.context_length}. "
                "The packed cache does not slide."
            )
        for (key_store, value_store), (key, value) in zip(
            self.layers, pairs, strict=True
        ):
            key_store.write(self._length, np.swapaxes(key, 2, 3))
            value_store.write(self._length, value)
        self._length += new

    def layer_float(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Decoded hub-layout ``(key, value)`` for debugging; not a memory-saving path."""
        key_store, value_store = self.layers[layer_idx]
        key = np.swapaxes(key_store.read(self._length), 2, 3)
        return np.ascontiguousarray(key), value_store.read(self._length)

    def memory_report(self, tokens: int | None = None) -> CacheMemoryReport:
        """Bytes used by ``tokens`` cached tokens (default: the current length)."""
        length = self._length if tokens is None else tokens
        if not 0 <= length <= self.context_length:
            raise ValueError(f"tokens={length} is outside [0, {self.context_length}].")
        payload = norms = raw = allocated = 0
        formula_bits = 0
        for store in (s for pair in self.layers for s in pair):
            arrays = store.arrays()
            allocated += sum(a.nbytes for a in arrays.values())
            if store.packed is not None and store.norms is not None:
                payload += store.packed[:, :, 0].nbytes * length
                norms += store.norms[:, :, 0].nbytes * length
                formula_bits += (
                    self.batch_size
                    * self.num_kv_heads
                    * length
                    * self.head_dim
                    * store.spec.bits
                )
            elif store.raw is not None:
                raw += store.raw[:, :, 0].nbytes * length
        return CacheMemoryReport(
            tokens=length,
            capacity=self.context_length,
            packed_payload_bytes=payload,
            norm_bytes=norms,
            baseline_float_bytes=raw,
            allocated_bytes=allocated,
            formula_payload_bytes=(formula_bits + 7) // 8,
        )

    def state_dict(self) -> dict[str, Any]:
        """Snapshot (copied buffers) tagged with the format and config hash."""
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "config_hash": self.config.config_hash(),
            "shape": [self.num_layers, self.num_kv_heads, self.head_dim],
            "context_length": self.context_length,
            "length": self._length,
            "layers": [
                {
                    "key": {n: a.copy() for n, a in k.arrays().items()},
                    "value": {n: a.copy() for n, a in v.arrays().items()},
                }
                for k, v in self.layers
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != FORMAT_NAME or (
            state.get("format_version") != FORMAT_VERSION
        ):
            raise ValueError(
                f"Cache state format {state.get('format')} v{state.get('format_version')} "
                f"is not {FORMAT_NAME} v{FORMAT_VERSION}."
            )
        if state.get("config_hash") != self.config.config_hash():
            raise ValueError(
                "Cache state was written with a different TurboQuant config."
            )
        if state.get("shape") != [
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        ] or (state.get("context_length") != self.context_length):
            raise ValueError("Cache state shape does not match this cache.")
        if not 0 <= int(state["length"]) <= self.context_length:
            raise ValueError(f"Cache state length {state['length']} is out of range.")
        layer_states = state.get("layers")
        if not isinstance(layer_states, list) or len(layer_states) != self.num_layers:
            raise ValueError("Cache state has the wrong number of layers.")

        # Validate everything before writing so a bad state never half-overwrites the cache.
        copies: list[tuple[np.ndarray, np.ndarray]] = []
        for (key_store, value_store), layer_state in zip(
            self.layers, layer_states, strict=True
        ):
            for store, kind in ((key_store, "key"), (value_store, "value")):
                arrays = layer_state.get(kind, {})
                expected = store.arrays()
                if set(arrays) != set(expected):
                    raise ValueError(
                        f"Cache state {kind} buffers do not match this config."
                    )
                for name, target in expected.items():
                    source = np.asarray(arrays[name])
                    if source.shape != target.shape or source.dtype != target.dtype:
                        raise ValueError(
                            f"Cache state buffer '{kind}/{name}' has wrong shape/dtype."
                        )
                    copies.append((target, source))
        for target, source in copies:
            target[...] = source
        self._length = int(state["length"])
