# SPDX-License-Identifier: BSD-3-Clause
"""Offline NIAH dataset invariants; no model or device execution."""

import importlib
import json
import re
import struct
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


class CharacterTokenizer:
    """Deterministic offset-aware fixture; special/chat text stays visible."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(c) for c in text]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict:
        assert return_offsets_mapping
        return {
            "input_ids": self.encode(text, add_special_tokens),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        assert add_generation_prompt and not enable_thinking
        text = "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages)
        text += "<assistant>\n<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_data")


@pytest.fixture
def tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer()


@pytest.fixture
def base(data: ModuleType) -> dict:
    text = "Natural English background words are preserved without padding. " * 120
    return {
        "base_id": "test_00",
        "split": "test",
        "project": "LARK",
        "expected_answer": "000042",
        "background": text,
        "background_sha256": data.text_digest(text),
    }


@pytest.mark.parametrize("depth", [0, 25, 50, 75, 100])
def test_full_chat_budget_unique_answer_and_needle_spans(
    data: ModuleType, tokenizer: CharacterTokenizer, base: dict, depth: int
) -> None:
    row, prompt, ids = data.make_case(tokenizer, base, 1024, depth)
    assert 0 <= 960 - len(ids) <= 16
    assert row["input_tokens"] == len(ids)
    assert row["conservative_max_total_tokens"] <= 1024
    assert prompt.count("000042") == 1
    begin, end = row["needle_char_span"]
    assert prompt[begin:end] == row["needle"]
    assert ids[row["needle_start_token"] : row["needle_end_token"]] == tokenizer.encode(
        row["needle"]
    )
    assert row["question_start_token"] >= row["needle_end_token"]
    assert row["actual_depth_percent"] == pytest.approx(depth, abs=2)
    assert row["final_prefill_chunk_start_token"] == max(0, len(ids) - 128)
    assert row["needle_overlaps_final_ar128"] == (
        row["needle_end_token"] > row["final_prefill_chunk_start_token"]
    )
    # No added attention padding or injected secret in the question.
    assert "000042" not in prompt[prompt.index("Question:") :]
    assert "<pad>" not in prompt


def test_repeatable_and_does_not_mutate_source(
    data: ModuleType, tokenizer: CharacterTokenizer, base: dict
) -> None:
    before = json.dumps(base, sort_keys=True)
    assert data.make_case(tokenizer, base, 1024, 25) == data.make_case(
        tokenizer, base, 1024, 25
    )
    assert json.dumps(base, sort_keys=True) == before


def test_empty_and_short_pilots_preserve_question(
    data: ModuleType, tokenizer: CharacterTokenizer, base: dict
) -> None:
    empty, text, _ = data.make_case(tokenizer, base, 1024, 50, no_haystack=True)
    short, short_text, _ = data.make_case(
        tokenizer, base, 1024, 50, short_haystack=True
    )
    assert empty["background_tokens"] == 0
    assert empty["input_tokens"] < short["input_tokens"]
    assert "Question:" in text and "Question:" in short_text
    assert empty["expected_answer"] == "000042"


def test_refuse_leakage_and_unfit_prompt(
    data: ModuleType, tokenizer: CharacterTokenizer, base: dict
) -> None:
    with pytest.raises(ValueError, match="leakage"):
        data.rendered_prompt(tokenizer, "Secret 000042", "LARK", "000042", 0)
    with pytest.raises(ValueError, match="intact chat prompt"):
        data.make_case(tokenizer, base, 128, 0)
    with pytest.raises(ValueError, match="Invalid context"):
        data.make_case(tokenizer, base, 64, 0)
    with pytest.raises(ValueError, match="Unsupported depth"):
        data.make_case(tokenizer, base, 1024, 7)


def test_document_boundaries_and_sources_are_preserved(data: ModuleType) -> None:
    rows = [
        "\n",
        " = First = \n",
        "Natural text\n",
        " == Section == \n",
        "Other text\n",
        " = Second = \n",
        "More.\n",
    ]
    docs = data.document_rows(rows)
    assert [d["title"] for d in docs] == ["= First =", "= Second ="]
    assert docs[0]["text"] == "".join(rows[1:5])
    assert docs[0]["sha256"] == data.text_digest(docs[0]["text"])


def natural_documents(data: ModuleType, count: int = 20) -> list[dict]:
    docs = []
    for i in range(count):
        text = f" = Document {i} = \n" + "A natural sentence describes the world. " * 60
        docs.append(
            {"title": f"Document {i}", "text": text, "sha256": data.text_digest(text)}
        )
    return docs


def test_base_seeds_disjoint_documents_and_six_digit_answers(
    data: ModuleType, tokenizer: CharacterTokenizer
) -> None:
    bases = data.base_examples(natural_documents(data), tokenizer, 1024, count=2)
    assert len(bases) == 4
    assert {b["split"] for b in bases} == {"test", "pilot", "smoke"}
    used = set()
    for base in bases:
        hashes = {d["sha256"] for d in base["source_documents"]}
        assert not used & hashes
        used.update(hashes)
        assert re.fullmatch(r"\d{6}", base["expected_answer"])
        assert base["expected_answer"] not in base["background"]
    assert len({b["expected_answer"] for b in bases}) == len(bases)
    assert len({b["seed"] for b in bases}) == len(bases)
    assert (
        data.base_examples(natural_documents(data), tokenizer, 1024, count=2) == bases
    )


def test_insufficient_source_fails_and_duplicates_are_deduplicated(
    data: ModuleType, tokenizer: CharacterTokenizer
) -> None:
    with pytest.raises(ValueError, match="Not enough"):
        data.base_examples(natural_documents(data, 1), tokenizer, 1024, count=2)
    docs = natural_documents(data, 20)
    assert data.base_examples(docs * 2, tokenizer, 1024, count=2) == data.base_examples(
        docs, tokenizer, 1024, count=2
    )


def test_freeze_once_manifest_binary_hash_and_crossed_design(
    data: ModuleType, tokenizer: CharacterTokenizer, tmp_path: Path
) -> None:
    root = tmp_path / "dataset"
    args = (
        root,
        tokenizer,
        natural_documents(data),
        {"source": "fixture"},
        {"tokenizer": "fixture"},
    )
    manifest = data.freeze_dataset(*args, capacities=(1024, 2048), base_count=2)
    assert manifest["case_counts"] == {"test": 20, "pilot": 2, "smoke": 2}
    assert json.loads((root / "data_manifest.json").read_text()) == manifest
    for case in manifest["cases"]:
        path = root / case["tokens_file"]
        raw = path.read_bytes()
        assert len(raw) == 4 * case["input_tokens"]
        ids = list(struct.unpack(f"<{case['input_tokens']}i", raw))
        assert ids == tokenizer.encode((root / case["prompt_file"]).read_text())
        assert data.digest(path) == case["tokens_sha256"]
        assert data.digest(root / case["prompt_file"]) == case["prompt_sha256"]
    test = [c for c in manifest["cases"] if c["split"] == "test"]
    assert len({c["case_id"] for c in test}) == 20
    assert len({c["base_id"] for c in test}) == 2
    with pytest.raises(FileExistsError):
        data.freeze_dataset(*args, capacities=(1024, 2048), base_count=2)


def test_bad_capacity_order_rejected_before_writes(
    data: ModuleType, tokenizer: CharacterTokenizer, tmp_path: Path
) -> None:
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="increasing"):
        data.freeze_dataset(root, tokenizer, [], {}, {}, capacities=(2048, 1024))
    assert not root.exists()


def test_chat_rendering_mismatch_is_not_accepted(data: ModuleType, base: dict) -> None:
    class WrongChatTokenizer(CharacterTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str | list[int]:
            result = super().apply_chat_template(*args, **kwargs)
            return [*result, 999] if isinstance(result, list) else result

    with pytest.raises(ValueError, match="differs"):
        data.make_case(WrongChatTokenizer(), base, 1024, 0)


def test_chat_template_can_return_batch_encoding(data: ModuleType, base: dict) -> None:
    class BatchTokenizer(CharacterTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str | dict:
            result = super().apply_chat_template(*args, **kwargs)
            return {"input_ids": result} if isinstance(result, list) else result

    record, _, ids = data.make_case(BatchTokenizer(), base, 1024, 0)
    assert record["input_tokens"] == len(ids)
