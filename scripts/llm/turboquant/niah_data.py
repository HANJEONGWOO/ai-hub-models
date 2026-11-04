# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Freeze a small, token-accounted single-needle retrieval benchmark.

This is a project-specific NIAH protocol, not an invocation of the original
NIAH suite. It neither selects quantizers nor uses model outputs to select
examples. A base example is crossed with every context capacity and depth;
therefore those derived prompts are not independent statistical samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CAPACITIES = (1024, 4096, 8192, 16384)
DEPTHS = (0, 25, 50, 75, 100)
TEST_SEED = 2026110201
PILOT_SEED = 2026110202
SMOKE_SEED = 2026110203
DEFAULT_CHECKPOINT = Path(
    "/home/hjw/.qaihm/qai-hub-models/models/qwen3_1_7b/v2/qwen3_1_7b_w4a16"
)
DEFAULT_SOURCE = (
    Path("/home/hjw/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots")
    / REVISION
    / "wikitext-2-raw-v1/train-00000-of-00001.parquet"
)
SYSTEM = "You are a helpful AI assistant. Follow the user's answer format."


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise ValueError("Tokenizer did not provide one offset per token")
    return ids, offsets


def document_rows(rows: list[str]) -> list[dict[str, str]]:
    """Preserve WikiText document text, including headings and punctuation."""
    result: list[dict[str, str]] = []
    title, current = "", []
    for row in rows:
        if re.fullmatch(r"\s*= [^=]+ =\s*", row):
            if current:
                body = "".join(current)
                result.append(
                    {"title": title, "text": body, "sha256": text_digest(body)}
                )
            title, current = row.strip(), [row]
        elif current:
            current.append(row)
    if current:
        body = "".join(current)
        result.append({"title": title, "text": body, "sha256": text_digest(body)})
    return result


def base_examples(
    documents: list[dict[str, str]],
    tokenizer: Any,
    max_capacity: int,
    count: int = 10,
    test_seed: int = TEST_SEED,
) -> list[dict[str, Any]]:
    """Allocate disjoint source documents before constructing any model input."""
    # Disjoint *documents* across base IDs; multiple depths/lengths reuse a base.
    # WikiText contains a small number of exactly duplicate/empty documents.
    # Content-hash de-duplication is a fixed source-only rule, not an accuracy
    # filter; a repeated document must not appear in different base examples.
    unique = {document["sha256"]: document for document in documents}
    docs = sorted(
        unique.values(),
        key=lambda d: text_digest(f"NIAH-background-v1:{test_seed}:{d['sha256']}"),
    )
    cursor, result, used_answers = 0, [], set()
    roles = [("test", test_seed, i) for i in range(count)]
    roles += [("pilot", PILOT_SEED, 0), ("smoke", SMOKE_SEED, 0)]
    for role, seed, index in roles:
        source_documents, text, token_count = [], "", 0
        # Full source documents, never repeated to fill a context.
        while token_count < max_capacity + 256:
            if cursor >= len(docs):
                raise ValueError(
                    "Not enough disjoint natural-text documents for protocol"
                )
            document = docs[cursor]
            cursor += 1
            source_documents.append({k: document[k] for k in ("title", "sha256")})
            text = "\n\n".join((text, document["text"])) if text else document["text"]
            token_count = len(encode(tokenizer, text))
        # Role/domain separation prevents adjacent root seeds from colliding
        # with another split's per-example seed.
        example_seed = int(
            text_digest(f"NIAH-secret-v1:{role}:{seed}:{index}")[:16], 16
        )
        rng = random.Random(example_seed)
        while True:
            project = "".join(
                rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(8)
            )
            answer = f"{rng.randrange(1_000_000):06d}"
            if (
                project not in text
                and answer not in text
                and answer not in used_answers
            ):
                break
        used_answers.add(answer)
        result.append(
            {
                "base_id": f"{role}_{index:02d}",
                "split": role,
                "seed": example_seed,
                "root_seed": seed,
                "project": project,
                "expected_answer": answer,
                "background": text,
                "background_tokens": token_count,
                "background_sha256": text_digest(text),
                "source_documents": source_documents,
            }
        )
    return result


def insert_needle(
    tokenizer: Any, background: str, needle: str, depth: int
) -> tuple[str, int, int, float]:
    """Insert at a word boundary nearest the requested background-token depth."""
    if depth not in DEPTHS:
        raise ValueError(f"Unsupported depth: {depth}")
    ids, offsets = with_offsets(tokenizer, background)
    if depth == 0 or not ids:
        cut = 0
    elif depth == 100:
        cut = len(background)
    else:
        target = round(len(ids) * depth / 100)
        target_char = offsets[min(target, len(offsets) - 1)][0]
        # A word is never split to insert the key/value fact.
        boundaries = [0] + [m.end() for m in re.finditer(r"\s+", background)]
        boundaries.append(len(background))
        cut = min(boundaries, key=lambda n: (abs(n - target_char), n))
    before, after = background[:cut], background[cut:]
    prefix = before + ("\n\n" if before else "")
    result = prefix + needle + ("\n\n" + after if after else "")
    actual = 100 * len(encode(tokenizer, before)) / len(ids) if ids else 0.0
    return result, len(prefix), len(prefix) + len(needle), actual


def rendered_prompt(
    tokenizer: Any, background: str, project: str, answer: str, depth: int
) -> tuple[str, list[int], dict[str, Any]]:
    needle = f"The access code for project {project} is {answer}."
    question = (
        f"What is the access code for project {project}? "
        "Return only the six-digit code, with no explanation."
    )
    if answer in background or answer in question:
        raise ValueError("Answer leakage outside the needle")
    context, begin, end, actual_depth = insert_needle(
        tokenizer, background, needle, depth
    )
    user_prefix = (
        "Read the following text and answer the question below.\n\n<context>\n"
    )
    user = user_prefix + context + "\n</context>\n\nQuestion: " + question
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids, offsets = with_offsets(tokenizer, text)
    chat_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if isinstance(chat_ids, Mapping):
        chat_ids = chat_ids["input_ids"]
    if ids != list(chat_ids):
        raise ValueError("Rendered text tokenization differs from apply_chat_template")
    if text.count(answer) != 1 or text.count(needle) != 1:
        raise ValueError("Needle or answer is not unique in the complete prompt")
    context_start = text.index(user) + len(user_prefix)
    needle_start, needle_end = context_start + begin, context_start + end
    token_span = [
        i for i, (a, b) in enumerate(offsets) if b > needle_start and a < needle_end
    ]
    if not token_span:
        raise ValueError("Needle has no tokenizer offsets")
    question_char = text.index(question, needle_end)
    question_token = next(i for i, (_, b) in enumerate(offsets) if b > question_char)
    start_token, end_token = min(token_span), max(token_span) + 1
    # Existing HTP runner uses a partial *first* prefill chunk, then full
    # AR128 chunks; the final chunk is therefore the last 128 prompt tokens.
    final_chunk = max(0, len(ids) - 128)
    return (
        text,
        ids,
        {
            "needle": needle,
            "needle_char_span": [needle_start, needle_end],
            "needle_start_token": start_token,
            "needle_end_token": end_token,
            "needle_token_span_end_exclusive": True,
            "question_start_token": question_token,
            "needle_to_question_tokens": question_token - end_token,
            "background_tokens": len(encode(tokenizer, background)),
            "actual_depth_percent": actual_depth,
            "final_prefill_chunk_start_token": final_chunk,
            "needle_overlaps_final_ar128": end_token > final_chunk,
            "needle_fully_in_final_ar128": start_token >= final_chunk,
            "prefill_ar": 128,
        },
    )


def make_case(
    tokenizer: Any,
    base: dict[str, Any],
    capacity: int,
    depth: int,
    max_new_tokens: int = 64,
    no_haystack: bool = False,
    short_haystack: bool = False,
) -> tuple[dict[str, Any], str, list[int]]:
    budget = capacity - max_new_tokens
    if capacity <= 0 or max_new_tokens <= 0 or budget <= 0:
        raise ValueError("Invalid context/generation budget")
    text = base["background"]
    ends = [m.end() for m in re.finditer(r"\S+", text)]
    if short_haystack:
        ends = ends[:64]
    if no_haystack:
        ends = []

    def candidate(words: int) -> tuple[str, list[int], dict[str, Any]]:
        background = text[: ends[words - 1]] if words else ""
        return rendered_prompt(
            tokenizer, background, base["project"], base["expected_answer"], depth
        )

    empty = candidate(0)
    if len(empty[1]) > budget:
        raise ValueError("Context budget cannot hold the intact chat prompt and needle")
    best, low, high, best_words = empty, 0, len(ends), 0
    # Binary search is only a length approximation (BPE length need not be
    # strictly monotonic). Every accepted candidate is checked in full; a small
    # local scan improves fill without ever adding synthetic padding tokens.
    while low <= high:
        middle = (low + high) // 2
        value = candidate(middle)
        if len(value[1]) <= budget:
            if len(value[1]) >= len(best[1]):
                best, best_words = value, middle
            low = middle + 1
        else:
            high = middle - 1
    for words in range(max(0, best_words - 4), min(len(ends), best_words + 12) + 1):
        value = candidate(words)
        if len(best[1]) < len(value[1]) <= budget:
            best, best_words = value, words
    prompt, ids, metadata = best
    if not no_haystack and not short_haystack and budget - len(ids) > 16:
        raise ValueError(
            "Natural-text prompt is more than 16 tokens below fixed budget"
        )
    suffix = "_empty" if no_haystack else ("_short" if short_haystack else "")
    case_id = f"{base['base_id']}_cl{capacity}_d{depth:03d}{suffix}"
    record = {
        "case_id": case_id,
        "base_id": base["base_id"],
        "split": base["split"],
        "capacity": capacity,
        "input_tokens": len(ids),
        "input_budget": budget,
        "input_budget_slack": budget - len(ids),
        "max_new_tokens": max_new_tokens,
        "conservative_max_total_tokens": len(ids) + max_new_tokens,
        "depth": depth,
        "project": base["project"],
        "expected_answer": base["expected_answer"],
        "background_sha256": base["background_sha256"],
        "used_background_characters": ends[best_words - 1] if best_words else 0,
        "tokens_file": f"assets/{case_id}.bin",
        "prompt_file": f"prompts/{case_id}.txt",
        "prompt_sha256": text_digest(prompt),
        **metadata,
    }
    return record, prompt, ids


def freeze_dataset(
    root: Path,
    tokenizer: Any,
    documents: list[dict[str, str]],
    source_identity: dict[str, Any],
    tokenizer_identity: dict[str, Any],
    capacities: tuple[int, ...] = CAPACITIES,
    base_count: int = 10,
) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite any existing data root: {root}")
    if not capacities or sorted(set(capacities)) != list(capacities) or base_count < 1:
        raise ValueError(
            "Expected increasing distinct capacities and positive base count"
        )
    bases = base_examples(documents, tokenizer, max(capacities), base_count)
    root.mkdir(parents=True, exist_ok=False)
    for sub in ("assets", "prompts", "backgrounds"):
        (root / sub).mkdir()
    cases, base_metadata = [], []
    for base in bases:
        background_file = f"backgrounds/{base['base_id']}.txt"
        with (root / background_file).open("x", encoding="utf-8") as f:
            f.write(base["background"])
        base_metadata.append(
            {
                **{k: v for k, v in base.items() if k != "background"},
                "file": background_file,
            }
        )
        if base["split"] == "pilot":
            settings = [
                (capacities[0], 50, True, False),
                (capacities[0], 50, False, True),
            ]
        elif base["split"] == "smoke":
            settings = [(capacity, 50, False, False) for capacity in capacities]
        else:
            settings = [
                (capacity, depth, False, False)
                for capacity in capacities
                for depth in DEPTHS
            ]
        for capacity, depth, empty, short in settings:
            record, prompt, ids = make_case(
                tokenizer,
                base,
                capacity,
                depth,
                no_haystack=empty,
                short_haystack=short,
            )
            raw = struct.pack(f"<{len(ids)}i", *ids)
            with (root / record["tokens_file"]).open("xb") as f:
                f.write(raw)
            with (root / record["prompt_file"]).open("x", encoding="utf-8") as f:
                f.write(prompt)
            record["tokens_sha256"] = hashlib.sha256(raw).hexdigest()
            cases.append(record)
    manifest = {
        "schema_version": 1,
        "benchmark": "single-needle exact-code retrieval, local frozen NIAH protocol",
        "source": source_identity,
        "tokenizer": tokenizer_identity,
        "protocol": {
            "capacities": list(capacities),
            "depths_percent": list(DEPTHS),
            "base_count": base_count,
            "test_seed": TEST_SEED,
            "pilot_seed": PILOT_SEED,
            "smoke_seed": SMOKE_SEED,
            "max_new_tokens": 64,
            "enable_thinking": False,
            "add_generation_prompt": True,
            "decoding": "greedy, stop at EOS or 64 generated tokens",
            "input_format": "little-endian signed int32 tokenizer IDs; no artificial attention padding tokens",
            "budget": "full rendered chat prompt <= capacity - 64; crop natural background at whole-word boundary only",
            "minimum_fill": "test/smoke prompt may be <=16 tokens below budget; actual count recorded",
            "selection": "deduplicate exact document content hashes, then salted source-document hash order, fixed seeds, no model outcomes",
            "independence": "source documents disjoint between base IDs; length/depth variants share each base, bootstrap by base_id",
            "scope": "WikiText background may overlap historical corpus or model pretraining; synthetic secret and task are new. No claim of unseen-corpus generalization.",
            "pilot": "two prompt-format checks plus one separate long smoke example per capacity; excluded from test accuracy",
            "primary_metric": "strip leading/trailing whitespace then exact equality with six-digit answer, preserving leading zeros",
            "secondary_metric": "exactly one standalone six-digit code in generated answer, equal to expected; repeated/listed codes fail",
        },
        "base_examples": base_metadata,
        "case_counts": {
            split: sum(c["split"] == split for c in cases)
            for split in ("test", "pilot", "smoke")
        },
        "cases": cases,
    }
    save_new(root / "data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--capacities", type=int, nargs="+", default=list(CAPACITIES))
    parser.add_argument("--base-count", type=int, default=10)
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError(args.root)
    import pandas as pd
    from transformers import AutoTokenizer

    checkpoint = args.checkpoint_dir.expanduser().resolve()
    source = args.source_parquet.expanduser().absolute()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer with offset mapping is required")
    documents = document_rows(pd.read_parquet(source)["text"].tolist())
    tokenizer_files = [
        p
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
        )
        if (p := checkpoint / name).is_file()
    ]
    source_revision = (
        source.parts[source.parts.index("snapshots") + 1]
        if "snapshots" in source.parts
        else "local-unversioned-file-see-sha256"
    )
    manifest = freeze_dataset(
        args.root.expanduser(),
        tokenizer,
        documents,
        {
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
            "official_split": source.name.split("-", 1)[0],
            "revision": source_revision,
            "path": str(source),
            "sha256": digest(source),
            "document_count": len(documents),
            "unique_document_hashes": len(
                {document["sha256"] for document in documents}
            ),
        },
        {
            "checkpoint_dir": str(checkpoint),
            "class": type(tokenizer).__name__,
            "files_sha256": {str(p): digest(p) for p in tokenizer_files},
            "chat_template_sha256": text_digest(tokenizer.chat_template),
            "eos_token_id": tokenizer.eos_token_id,
        },
        tuple(args.capacities),
        args.base_count,
    )
    print(
        json.dumps(
            {
                "root": str(args.root),
                "case_counts": manifest["case_counts"],
                "manifest_sha256": digest(args.root / "data_manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
