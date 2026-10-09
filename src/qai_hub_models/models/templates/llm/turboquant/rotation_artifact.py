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
LAYER_FORMAT = "qaihm-k-layer-dense-rotation-v1"


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
    if path is None:
        return config
    data = json.loads(Path(path).read_text())
    if data.get("format") == LAYER_FORMAT:
        return replace(config, key_layers=load_layer_rotations(path, config.key))
    return replace(config, key=replace(config.key, dense_matrix=load_rotation(path)))


def load_layer_rotations(path: str | Path, base: Any) -> tuple:
    data = json.loads(Path(path).read_text())
    if (
        data.get("format") != LAYER_FORMAT
        or data.get("dimension") != 128
        or data.get("dtype") != "float32"
        or data.get("sharing") != "per_layer_all_kv_heads"
        or data.get("convention") != "row_vector @ R.T"
    ):
        raise ValueError("Unsupported layer rotation artifact contract")
    seeds = data["layer_seeds"]
    if (
        not seeds
        or any(type(s) is not int for s in seeds)
        or set(map(str, seeds)) != set(data["matrices"])
    ):
        raise ValueError("Layer rotation mapping is incomplete")
    matrices = {}
    for seed, item in data["matrices"].items():
        matrix = validate_matrix(item["matrix"], 128)
        if matrix_digest(matrix) != item["matrix_f32_sha256"]:
            raise ValueError("Layer matrix checksum mismatch")
        matrices[int(seed)] = tuple(float(x) for x in matrix.ravel())
    identity = {
        "layer_seeds": seeds,
        "matrices": {
            s: data["matrices"][s]["matrix_f32_sha256"]
            for s in sorted(data["matrices"])
        },
    }
    checksum = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if checksum != data["policy_sha256"]:
        raise ValueError("Layer rotation policy checksum mismatch")
    return tuple(replace(base, seed=s, dense_matrix=matrices[s]) for s in seeds)


def save_layer_rotations(
    path: Path, seeds: list[int], matrices: dict[int, np.ndarray], provenance: dict
) -> dict:
    values = {str(s): validate_matrix(matrices[s], 128) for s in sorted(set(seeds))}
    identity = {
        "layer_seeds": seeds,
        "matrices": {s: matrix_digest(m) for s, m in values.items()},
    }
    data = {
        "format": LAYER_FORMAT,
        "dimension": 128,
        "dtype": "float32",
        "sharing": "per_layer_all_kv_heads",
        "convention": "row_vector @ R.T",
        "layer_seeds": seeds,
        "matrices": {
            s: {"matrix": m.tolist(), "matrix_f32_sha256": matrix_digest(m)}
            for s, m in values.items()
        },
        "policy_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "provenance": provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
    return data


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
