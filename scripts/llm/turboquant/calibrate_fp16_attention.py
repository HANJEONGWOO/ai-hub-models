# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Calibrate only integer activations on the FP16 KV/QK/AV path.

Stages: prepare (train-only dataset + immutable source identity), calibrate
(one split part at a time), finalize (fail-closed completeness checks).
Intermediate hidden states are disk-backed; all internal activations are not
retained. The default is conventional global pass-through activation min/max
calibration with frozen weight QDQ and explicit FP16 cache/attention rounding.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
from convert_parts import input_shape

from qai_hub_models.models.templates.llm.turboquant.calibration import (
    create_sim,
    digest,
    finish_observers,
    parameter_interfaces,
    preserve_parameter_interfaces,
    range_session,
    start_observers,
    validate_encodings,
)
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.fp16_attention import (
    use_fp16_kv_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)

PROFILE = "baseline_fp16_kv_fp16_attn_calibrated"
CONTEXT = 1024
AR = 128


def write_json(path: Path, data: dict) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def schedule() -> Iterator[tuple[str, int, int, int, int]]:
    """Eight prefill chunks + eight AR1 probes, last probe commits token1023.

    Intermediate probes do not commit KV: the following prefill includes that
    same next token. Thus probe and chunk executions see exactly the same prefix.
    """
    for chunk in range(8):
        start = chunk * AR
        count = 127 if chunk == 7 else AR
        yield "prefill", chunk, start, count, AR
        yield "decode", chunk, start + count, 1, 1


def prepare(args: argparse.Namespace) -> None:
    from types import SimpleNamespace

    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    from qai_hub_models.models.templates.lm_driver.utils.rope_embedding import (
        RopeEmbedding,
    )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "experiment.json").exists():
        raise FileExistsError("Use an unused calibration experiment directory")
    split = read(args.split_dir / "split_manifest.json")
    if split["model_id"] != "qwen3_1_7b" or len(split["parts"]) != 4:
        raise ValueError("This calibration experiment is restricted to Qwen3-1.7B")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    raw = Path(
        hf_hub_download(
            repo_id="Salesforce/wikitext",
            repo_type="dataset",
            filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
        )
    )
    # Match WikiText(DatasetSplit.TRAIN), including add_special_tokens=True.
    separator = tokenizer.bos_token if tokenizer.bos_token is not None else "\n\n"
    token_ids = np.asarray(
        tokenizer(
            separator.join(pd.read_parquet(raw)["text"].tolist()),
            add_special_tokens=True,
        ).input_ids,
        dtype=np.int32,
    )
    available = len(token_ids) // CONTEXT
    indices = np.random.default_rng(args.seed).choice(
        available, args.samples, replace=False
    )
    tokens = np.stack([token_ids[i * CONTEXT : (i + 1) * CONTEXT] for i in indices])
    np.save(out / "train_tokens.npy", tokens, allow_pickle=False)
    config = AutoConfig.from_pretrained(args.checkpoint)
    rope = RopeEmbedding(model=SimpleNamespace(config=config), context_length=CONTEXT)
    np.save(
        out / "rope_cos.npy",
        rope.cos[0, 0].numpy().astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        out / "rope_sin.npy",
        rope.sin[0, 0].numpy().astype(np.float32),
        allow_pickle=False,
    )
    sources = []
    for name, part in split["parts"].items():
        source = Path(part["bundle_dir"]).resolve() / (part["class"] + ".onnx")
        sources.append(
            {
                "part": name,
                "source": str(source),
                "source_onnx_sha256": digest(source),
                "source_encodings_sha256": digest(source.with_suffix(".encodings")),
                "source_data_sha256": {
                    p.name: digest(p) for p in sorted(source.parent.glob("*.data"))
                },
            }
        )
    experiment = {
        "profile": PROFILE,
        "model_id": "qwen3_1_7b",
        "context_length": CONTEXT,
        "samples": args.samples,
        "seed": args.seed,
        "batch_size": 1,
        "dataset": {
            "repository": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "train",
            "file": str(raw),
            "sha256": digest(raw),
            "available_windows": available,
            "window_indices": indices.tolist(),
            "separator": separator,
            "add_special_tokens": True,
        },
        "tokenizer": {
            p.name: digest(p) for p in Path(args.checkpoint).glob("*token*.json")
        },
        "inputs_sha256": {p.name: digest(p) for p in out.glob("*.npy")},
        "pad_token": tokenizer.pad_token_id,
        "mask_min": -100.0,
        "schedule": list(schedule()),
        "sources": sources,
        "fixed_parameter_interfaces": parameter_interfaces(
            [Path(s["source"]) for s in sources]
        ),
        "algorithm": "global_minmax_pass_through",
        "observer_policy": "activation_observers_pass_through; weight_QDQ_frozen; explicit_FP16_casts_active",
        "host_math": "ORT CUDA, TF32 disabled; hardware accumulation equivalence not claimed",
    }
    write_json(out / "experiment.json", experiment)
    print(json.dumps(experiment, indent=2), flush=True)


def target_graph(source: Path) -> tuple[onnx.ModelProto, dict]:
    model = onnx.load(source, load_external_data=False)
    enc = read(source.with_suffix(".encodings"))
    if any(v.name.startswith("past_") for v in model.graph.input):
        config = get_profile(PROFILE)
        result = use_fp16_kv_attention(
            apply_kv_profile(model, enc, config, AR, CONTEXT), config
        )
        # For this uncompressed profile the actual graph is sequence-dynamic;
        # prove AR1 and AR128 surgery are identical before using one observer.
        other = use_fp16_kv_attention(
            apply_kv_profile(model, enc, config, 1, CONTEXT), config
        )
        if (
            result.model.SerializeToString() != other.model.SerializeToString()
            or result.encodings != other.encodings
        ):
            raise ValueError(
                "AR1/AR128 graphs differ; shared dynamic calibration is unsafe"
            )
        model, enc = result.model, result.encodings
    # Infer while weights are external: part4 exceeds protobuf's 2GiB limit.
    return onnx.shape_inference.infer_shapes(model), enc


def verify_sources(experiment: dict) -> None:
    for item in experiment["sources"]:
        source = Path(item["source"])
        if (
            digest(source) != item["source_onnx_sha256"]
            or digest(source.with_suffix(".encodings"))
            != item["source_encodings_sha256"]
        ):
            raise ValueError("Original source graph or encodings changed")
        for name, sha in item["source_data_sha256"].items():
            if digest(source.parent / name) != sha:
                raise ValueError("Original weights changed")


def calibrate(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    exp = read(out / "experiment.json")
    verify_sources(exp)
    for name, sha in exp["inputs_sha256"].items():
        if digest(out / name) != sha:
            raise ValueError("Calibration inputs changed")
    info = exp["sources"][args.part - 1]
    source = Path(info["source"])
    part_dir = out / (
        f"part{args.part}" + (f"_{args.attempt_label}" if args.attempt_label else "")
    )
    part_dir.mkdir(exist_ok=False)
    model, enc = target_graph(source)
    # Original files remain external/read-only; the simulation owns only a copy.
    onnx.external_data_helper.load_external_data_for_model(model, str(source.parent))
    dummy = {}
    for v in model.graph.input:
        dims = [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
        shape = input_shape(v.name, dims, 1, CONTEXT)
        dummy[v.name] = np.zeros(
            shape,
            dtype=onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type),
        )
    # RoPE prevents zero-denominator dummy paths if AIMET needs runtime inference.
    if "position_ids_cos" in dummy:
        dummy["position_ids_cos"].fill(1)
    input_names = [v.name for v in model.graph.input]
    output_names = [v.name for v in model.graph.output]
    hidden_in = next(
        (
            n
            for n in input_names
            if n
            not in (
                "input_ids",
                "attention_mask",
                "position_ids_cos",
                "position_ids_sin",
            )
            and not n.startswith("past_")
        ),
        None,
    )
    hidden_out = next(n for n in output_names if not n.startswith("past_"))
    cache_shapes = {n: a.shape for n, a in dummy.items() if n.startswith("past_")}
    sim = create_sim(model, enc, dummy, part_dir / "sim")
    print(
        f"part{args.part}: simulation ready; {len(enc['activation_encodings'])} activation boundaries",
        flush=True,
    )
    tokens = np.load(out / "train_tokens.npy", mmap_mode="r")
    rope = {kind: np.load(out / f"rope_{kind}.npy") for kind in ("cos", "sin")}
    previous = None
    if hidden_in:
        prior = out / f"part{args.part - 1}"
        prior_report = read(prior / "report.json")
        if not prior_report["complete"]:
            raise ValueError("Upstream part has not completed quantized replay")
        previous = {
            k: np.load(prior / f"{k}_hidden.npy", mmap_mode="r")
            for k in ("prefill", "decode")
        }
    observer = (
        range_session(sim, enc, part_dir / "range_observers")
        if args.native_observers
        else None
    )
    outputs = [o.name for o in sim.session.get_outputs()]
    # AIMET appends _updated to quantized graph outputs (including disabled ones).
    output_map = {
        n: (n + "_updated" if n + "_updated" in outputs else n) for n in output_names
    }
    requested = [output_map[n] for n in output_names]
    reports = []
    start_observers(sim, enc)
    bounds = np.full(
        (len(enc["activation_encodings"]), 2), [np.inf, -np.inf], dtype=np.float32
    )
    global_observers = exp["algorithm"] == "global_minmax_pass_through"
    phases = ["observe"] if args.probe or global_observers else ["observe", "replay"]
    relay_phase = "observe" if global_observers else "replay"
    result = None
    for phase in phases:
        begin = time.monotonic()
        relay = None
        if phase == relay_phase and args.part < 4 and not args.probe:
            hidden_size = exp.get("hidden_size", 2048)
            relay = {
                k: np.lib.format.open_memmap(
                    part_dir / f"{k}_hidden.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(tokens), 8, ar, hidden_size),
                )
                for k, ar in (("prefill", AR), ("decode", 1))
            }
        total_steps = 0
        max_abs = 0.0
        count = min(1, len(tokens)) if args.probe else len(tokens)
        for window in range(count):
            cache = {
                n: np.zeros(
                    (*shape[:2], 128, 0) if "past_key_" in n else (*shape[:2], 0, 128),
                    dtype=np.float16,
                )
                for n, shape in cache_shapes.items()
            }
            for kind, chunk, pos, valid, ar in schedule():
                data = {}
                if hidden_in:
                    data[hidden_in] = np.asarray(previous[kind][window, chunk])[
                        None, ...
                    ]
                else:
                    ids = np.full((1, ar), exp["pad_token"], dtype=np.int32)
                    ids[0, ar - valid :] = tokens[window, pos : pos + valid]
                    data["input_ids"] = ids
                if "attention_mask" in input_names:
                    mask = np.full(
                        (1, 1, ar, CONTEXT), exp["mask_min"], dtype=np.float32
                    )
                    for row in range(ar):
                        mask[..., row, CONTEXT - ar - pos : CONTEXT - ar] = 0
                        mask[..., row, CONTEXT - valid : CONTEXT - ar + row + 1] = 0
                    data["attention_mask"] = mask
                    positions = np.concatenate(
                        (
                            np.full(ar - valid, max(pos - 1, 0)),
                            np.arange(pos, pos + valid),
                        )
                    ).astype(np.int64)
                    for name in ("cos", "sin"):
                        data[f"position_ids_{name}"] = rope[name][positions][
                            None, None, ...
                        ]
                for name, stored in cache.items():
                    axis = 3 if "past_key_" in name else 2
                    shape = list(stored.shape)
                    shape[axis] = CONTEXT - ar
                    value = np.zeros(shape, dtype=np.float16)
                    if stored.shape[axis]:
                        slices = [slice(None)] * 4
                        slices[axis] = slice(CONTEXT - ar - pos, None)
                        value[tuple(slices)] = stored
                    data[name] = value
                if phase == "observe" and observer is not None:
                    names = output_names + (
                        ["calibration_ranges"] if len(bounds) else []
                    )
                    values = observer.run(names, data)
                    if len(bounds):
                        ranges = values.pop().reshape(-1, 2)
                        if not np.isfinite(ranges).all():
                            raise ValueError("Nonfinite activation range")
                        bounds[:, 0] = np.minimum(bounds[:, 0], ranges[:, 0])
                        bounds[:, 1] = np.maximum(bounds[:, 1], ranges[:, 1])
                else:
                    values = sim.session.run(requested, data)
                if args.probe and observer is not None:
                    expected = sim.session.run(requested, data)
                    for a, b in zip(values, expected, strict=True):
                        np.testing.assert_allclose(a, b, atol=2e-3, rtol=2e-3)
                actual = dict(zip(output_names, values, strict=True))
                for name, value in actual.items():
                    if not np.isfinite(value).all():
                        raise ValueError(
                            f"Nonfinite output part{args.part}/{phase}/{window}/{kind}/{pos}/{name}"
                        )
                max_abs = max(max_abs, float(np.max(np.abs(actual[hidden_out]))))
                if relay is not None:
                    relay[kind][window, chunk] = actual[hidden_out][0]
                if kind == "prefill" or chunk == 7:
                    for name, cached in cache.items():
                        value = actual[name.removesuffix("_in") + "_out"]
                        if value.dtype != np.float16:
                            raise ValueError("Cache output lost FP16 storage")
                        axis = 3 if "past_key_" in name else 2
                        slices = [slice(None)] * 4
                        slices[axis] = slice(ar - valid, None)
                        cache[name] = np.concatenate(
                            (cached, value[tuple(slices)]), axis=axis
                        )
                total_steps += 1
                if args.probe and total_steps == 2:
                    break
            if window % 4 == 0 or window == count - 1:
                print(
                    f"part{args.part} {phase}: {window + 1}/{count} windows, {total_steps} steps, {time.monotonic() - begin:.1f}s",
                    flush=True,
                )
        if relay is not None:
            for value in relay.values():
                value.flush()
        reports.append(
            {
                "phase": phase,
                "windows": count,
                "steps": total_steps,
                "seconds": time.monotonic() - begin,
                "finite_outputs": True,
                "max_abs_hidden_or_logits": max_abs,
            }
        )
        if phase == "observe":
            if observer is not None:
                for e, limits in zip(enc["activation_encodings"], bounds, strict=True):
                    sim.qc_quantize_op_dict[e["name"]].update_encoding_stats(limits)
                np.save(part_dir / "activation_minmax.npy", bounds, allow_pickle=False)
            result, changes = finish_observers(sim, enc)
            result, fixed_interfaces = preserve_parameter_interfaces(
                enc, result, exp.get("fixed_parameter_interfaces", {})
            )
            changes = validate_encodings(enc, result)
            write_json(part_dir / "calibrated.encodings", result)
            write_json(part_dir / "encoding_changes.json", changes)
    # Original parameter encoding dictionaries are never re-exported by AIMET.
    assert result is not None
    validate_encodings(enc, result)
    verify_sources(exp)
    report = {
        **info,
        "complete": not args.probe,
        "probe_only": args.probe,
        "phases": reports,
        "encodings_file": f"part{args.part}/calibrated.encodings",
        "encodings_sha256": digest(part_dir / "calibrated.encodings"),
        "parameter_encodings_unchanged": result["param_encodings"]
        == enc["param_encodings"],
        "activation_changes": changes,
        "source_initializers_unchanged": True,
        "fixed_parameter_interfaces": fixed_interfaces,
    }
    report["observer_backend"] = (
        "onnx_reduce_minmax" if observer is not None else "aimet_updateStats"
    )
    report["versions"] = {
        package: importlib.metadata.version(package)
        for package in ("aimet-onnx", "onnx", "onnxruntime-gpu", "numpy")
    }
    report["implementation_sha256"] = {
        "driver": digest(Path(__file__)),
        "calibration": digest(Path(sys.modules[create_sim.__module__].__file__)),
    }
    write_json(part_dir / "report.json", report)
    print(f"part{args.part}: {'PROBE ONLY' if args.probe else 'complete'}", flush=True)


def finalize(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    exp = read(out / "experiment.json")
    verify_sources(exp)
    parts = {}
    fixed = parameter_interfaces([Path(s["source"]) for s in exp["sources"]])
    if exp.get("fixed_parameter_interfaces") != fixed:
        raise ValueError("Missing fixed parameter interfaces; use constrain-interfaces")
    for part, info in enumerate(exp["sources"], 1):
        report = read(out / f"part{part}" / "report.json")
        if not report["complete"] or report["probe_only"]:
            raise ValueError("Cannot finalize partial/probe calibration")
        expected_phases = (
            ["observe"]
            if exp["algorithm"] == "global_minmax_pass_through"
            else ["observe", "replay"]
        )
        if [p["phase"] for p in report["phases"]] != expected_phases or any(
            p["windows"] != exp["samples"]
            or p["steps"] != exp["samples"] * 16
            or not p["finite_outputs"]
            for p in report["phases"]
        ):
            raise ValueError("Incomplete calibration/replay data coverage")
        if not report["parameter_encodings_unchanged"]:
            raise ValueError("Parameter encodings changed")
        if report["encodings_sha256"] != digest(out / report["encodings_file"]):
            raise ValueError("Calibrated encoding changed")
        _, original = target_graph(Path(info["source"]))
        encodings = read(out / report["encodings_file"])
        constrained, _ = preserve_parameter_interfaces(original, encodings, fixed)
        if constrained != encodings:
            raise ValueError("Unconstrained parameter-bound activation interface")
        parts[Path(info["source"]).stem] = report
    manifest = {
        "status": "complete",
        "profile": PROFILE,
        "context_length": CONTEXT,
        "experiment_sha256": digest(out / "experiment.json"),
        "experiment": exp,
        "parts": parts,
    }
    write_json(out / "calibration_manifest.json", manifest)
    print("Calibration complete", flush=True)


def constrain_interfaces(args: argparse.Namespace) -> None:
    """Derive a deployment artifact from existing observations; no model rerun."""
    source = args.from_calibration.resolve()
    parent = read(source / "calibration_manifest.json")
    if parent["status"] != "complete":
        raise ValueError("Only complete observations can be constrained")
    exp = copy.deepcopy(parent["experiment"])
    verify_sources(exp)
    exp["fixed_parameter_interfaces"] = parameter_interfaces(
        [Path(s["source"]) for s in exp["sources"]]
    )
    exp["derivation"] = {
        "parent": str(source),
        "parent_manifest_sha256": digest(source / "calibration_manifest.json"),
        "reason": "Preserve frozen parameter-derived partition interfaces; no new observations or test-data tuning",
        "implementation_sha256": digest(Path(__file__)),
    }
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, sha in exp["inputs_sha256"].items():
        if digest(source / name) != sha:
            raise ValueError("Parent input changed")
        (out / name).symlink_to(source / name)
    write_json(out / "experiment.json", exp)
    for number, info in enumerate(exp["sources"], 1):
        prior = source / f"part{number}"
        report = copy.deepcopy(read(prior / "report.json"))
        path = prior / "calibrated.encodings"
        if digest(path) != report["encodings_sha256"]:
            raise ValueError("Parent encodings changed")
        _, before = target_graph(Path(info["source"]))
        after, kept = preserve_parameter_interfaces(
            before, read(path), exp["fixed_parameter_interfaces"]
        )
        part = out / f"part{number}"
        part.mkdir()
        write_json(part / "calibrated.encodings", after)
        changes = validate_encodings(before, after)
        write_json(part / "encoding_changes.json", changes)
        for data in prior.glob("*.npy"):
            (part / data.name).symlink_to(data)
        report.update(
            encodings_sha256=digest(part / "calibrated.encodings"),
            activation_changes=changes,
            fixed_parameter_interfaces=kept,
            observation_source=str(prior),
        )
        write_json(part / "report.json", report)
    finalize(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "calibrate", "finalize", "constrain-interfaces")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--from-calibration", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--part", type=int, choices=range(1, 5))
    parser.add_argument(
        "--probe",
        action="store_true",
        help="One-window tool compatibility check, never finalized",
    )
    parser.add_argument(
        "--native-observers",
        action="store_true",
        help="Equivalent GPU-native min/max reductions; preserve original/export weights",
    )
    parser.add_argument(
        "--attempt-label",
        choices=("native_probe", "native_probe_v2", "native_probe_v3"),
        help="Keep a compatibility probe separate from formal calibration",
    )
    args = parser.parse_args()
    if args.stage == "prepare" and (not args.split_dir or not args.checkpoint):
        parser.error("prepare requires --split-dir and --checkpoint")
    if args.stage == "calibrate" and args.part is None:
        parser.error("calibrate requires --part")
    if args.stage == "constrain-interfaces" and not args.from_calibration:
        parser.error("constrain-interfaces requires --from-calibration")
    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.attempt_label and not args.probe:
        parser.error("--attempt-label is for compatibility probes only")
    {
        "prepare": prepare,
        "calibrate": calibrate,
        "finalize": finalize,
        "constrain-interfaces": constrain_interfaces,
    }[args.stage](args)


if __name__ == "__main__":
    main()
