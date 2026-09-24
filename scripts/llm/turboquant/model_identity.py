# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Model/checkpoint identity shared by split, conversion and device assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

QWEN_SHAPES = {
    "qwen3_1_7b": (28, 2048, 16, 8, 128),
    "qwen3_4b": (36, 2560, 32, 8, 128),
}


def validate_model(identity: dict[str, Any]) -> None:
    if identity["model_id"] not in QWEN_SHAPES:
        return
    architecture = identity["architecture"]
    actual = tuple(
        architecture[k]
        for k in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    )
    if actual != QWEN_SHAPES[identity["model_id"]]:
        raise ValueError(
            f"Checkpoint architecture does not match {identity['model_id']}: {actual}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(checkpoint: Path, model_id: str) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    architecture = {
        key: config[key]
        for key in (
            "model_type",
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
        )
    }
    # Qwen3-4B uses 128, not hidden_size / num_attention_heads (80).
    architecture["head_dim"] = config.get(
        "head_dim", config["hidden_size"] // config["num_attention_heads"]
    )
    identity = {
        "model_id": model_id,
        "architecture": architecture,
        "config_tokenizer_sha256": {
            name: sha256_file(checkpoint / name)
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
            if (checkpoint / name).is_file()
        },
    }
    validate_model(identity)
    return identity
