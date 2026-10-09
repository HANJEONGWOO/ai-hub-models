# SPDX-License-Identifier: BSD-3-Clause
"""Content-addressed, opt-in K-only Dense rotation artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig

FORMAT = "qaihm-k-dense-rotation-v1"


def matrix_digest(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def validate_matrix(values: Any, dimension: int) -> np.ndarray:
    matrix = np.asarray(values, dtype="<f4")
    if matrix.size != dimension * dimension or not np.isfinite(matrix).all():
        raise ValueError("Invalid rotation shape or nonfinite coefficient")
    matrix = matrix.reshape(dimension, dimension)
    error = np.linalg.norm(
        matrix.astype(np.float64).T @ matrix - np.eye(dimension), ord=2
    )
    if error > 2e-6:
        raise ValueError(f"FP32 rotation is not orthogonal: spectral error {error}")
    matrix.setflags(write=False)
    return matrix


def load_rotation(path: str | Path) -> tuple[float, ...]:
    data = json.loads(Path(path).read_text())
    if (
        data["format"] != FORMAT
        or data["dtype"] != "float32"
        or data["sharing"] != "all_layers_all_kv_heads"
        or data.get("convention") != "row_vector @ R.T"
        or data["dimension"] != 128
    ):
        raise ValueError("Unsupported rotation artifact contract")
    matrix = validate_matrix(data["matrix"], data["dimension"])
    if matrix_digest(matrix) != data["matrix_f32_sha256"]:
        raise ValueError("Rotation artifact checksum mismatch")
    return tuple(float(x) for x in matrix.ravel())


def with_key_rotation(
    config: TurboQuantConfig, path: str | Path | None
) -> TurboQuantConfig:
    return (
        config
        if path is None
        else replace(config, key=replace(config.key, dense_matrix=load_rotation(path)))
    )


def save_rotation(path: Path, matrix: np.ndarray, provenance: dict) -> dict:
    matrix = validate_matrix(matrix, matrix.shape[0])
    data = {
        "format": FORMAT,
        "dimension": matrix.shape[0],
        "dtype": "float32",
        "sharing": "all_layers_all_kv_heads",
        "convention": "row_vector @ R.T",
        "matrix_f32_sha256": matrix_digest(matrix),
        "matrix": matrix.tolist(),
        "provenance": provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
    return data
