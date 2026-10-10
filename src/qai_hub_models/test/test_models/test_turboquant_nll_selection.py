# SPDX-License-Identifier: BSD-3-Clause
"""Document-level seed selection, disjoint folds and paired uncertainty."""

import importlib
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def experiment() -> Iterator[ModuleType]:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    sys.path.insert(0, str(scripts))
    try:
        yield importlib.import_module("rotation_nll_selection")
    finally:
        sys.path.remove(str(scripts))


def test_canonical_titles_handle_old_manifest_formats(experiment: ModuleType) -> None:
    assert experiment.canonical_title("= Head VI =") == experiment.canonical_title(
        "Head VI"
    )
    assert experiment.canonical_title(" =  Fort  Scott = ") == "fort scott"


def test_old_token_overlap_is_contiguous_and_ordered(experiment: ModuleType) -> None:
    old = experiment.grams(np.arange(30))
    assert not old.isdisjoint(experiment.grams(np.arange(10, 26)))
    assert old.isdisjoint(experiment.grams(np.arange(25, 9, -1)))
    assert not experiment.grams(np.arange(15))


def test_selection_exact_ties_choose_lowest_seed(experiment: ModuleType) -> None:
    assert experiment.choose_seed(np.ones((3, 4)), np.ones(4), [48, 42, 49]) == 42


def test_selection_uses_nll_not_mean_document_ppl(experiment: ModuleType) -> None:
    # Seed42 has lower total NLL, despite larger arithmetic document PPL.
    table = np.array([[0.0, 8.0], [4.1, 4.1]])
    assert experiment.choose_seed(table, np.ones(2), [42, 48]) == 42
    assert np.exp(table[0]).mean() > np.exp(table[1]).mean()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_candidate_loss_rejected(experiment: ModuleType, bad: float) -> None:
    with pytest.raises(ValueError, match="Invalid candidate"):
        experiment.choose_seed(np.array([[bad]]), np.array([1]), [42])


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_invalid_counts_rejected(experiment: ModuleType, bad: float) -> None:
    with pytest.raises(ValueError, match="Invalid candidate"):
        experiment.choose_seed(np.ones((1, 1)), np.array([bad]), [42])


def test_folds_fixed_balanced_and_disjoint(experiment: ModuleType) -> None:
    folds = experiment.fold_assignment(16)
    assert folds == experiment.fold_assignment(16)
    assert all(folds.count(i) == 4 for i in range(4))
    counts = np.ones(16)
    table = np.array([[2.0] * 16, [1.0] * 16])
    result = experiment.fold_analysis(table, counts, [42, 48], folds)
    assert result["selection_counts"] == {"42": 0, "48": 4}
    assert result["out_of_fold_mean_nll"] == 1
    assert result["out_of_fold_delta_vs_A"] == -1
    for fold in result["folds"]:
        assert len(fold["train_documents"]) == 12
        assert len(fold["excluded_documents"]) == 4
        assert set(fold["train_documents"]).isdisjoint(fold["excluded_documents"])


def test_excluded_fold_does_not_choose_its_seed(experiment: ModuleType) -> None:
    folds = np.repeat(np.arange(4), 4).tolist()
    table = np.array([[1.0] * 16, [2.0] * 16])
    table[1, :4] = 0
    result = experiment.fold_analysis(table, np.ones(16), [42, 48], folds)
    assert result["folds"][0]["selected_seed"] == 42
    assert result["folds"][0]["mean_nll"] == 1


def test_identical_alias_bootstrap_is_exact_zero(experiment: ModuleType) -> None:
    values = np.arange(1, 17, dtype=float)
    result = experiment.paired_bootstrap(values, values, np.ones(16))
    assert result["ci95"] == [0, 0]
    assert result["mean_nll_delta"] == 0
    assert result["equal_documents"] == 16
    assert result["top3_share_of_positive_nll_gains"] is None
    assert result["largest_improvement_document"] is None
    assert result["largest_regression_document"] is None


def test_paired_bootstrap_preserves_pairing(experiment: ModuleType) -> None:
    baseline = np.arange(1, 17, dtype=float) * 100
    candidate = baseline - 1
    result = experiment.paired_bootstrap(candidate, baseline, np.ones(16))
    assert result["ci95"] == [-1, -1]
    assert result["ci_entirely_below_zero"]
    assert result["improved_documents"] == 16
    assert result["ppl_relative_change"] == pytest.approx(math.expm1(-1))
    assert result == experiment.paired_bootstrap(candidate, baseline, np.ones(16))


def test_paired_bootstrap_uncertain_mixed_documents(experiment: ModuleType) -> None:
    baseline = np.full(16, 10.0)
    candidate = baseline + np.tile([-1, 1], 8)
    result = experiment.paired_bootstrap(candidate, baseline, np.ones(16))
    assert result["ci_contains_zero"]
    assert result["improved_documents"] == result["worse_documents"] == 8
    assert result["top3_share_of_positive_nll_gains"] == 3 / 8


def test_frozen_scope(experiment: ModuleType) -> None:
    p = experiment.PROTOCOL
    assert p["seeds"] == list(range(42, 50))
    assert p["validation_documents"] == p["heldout_documents"] == 16
    assert p["score_repeats"] == 1
    assert p["performance_repeats"] == 3
    assert p["bootstrap_replicates"] == 20000


def test_refuse_overwrite(experiment: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    experiment.write_json(path, {"seed": 42})
    with pytest.raises(FileExistsError):
        experiment.write_json(path, {"seed": 48})
    assert experiment.read(path) == {"seed": 42}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rms_norm_node___123", "rms_norm_node___GENERATED_ID"),
        ("rms_norm_node___158339__707022", "rms_norm_node___GENERATED_ID"),
        ("rms_norm_node___123__456__789", "rms_norm_node___GENERATED_ID"),
        ("rms_norm_node___123_output", "rms_norm_node___123_output"),
        ("different_node___123", "different_node___123"),
    ],
)
def test_only_random_rms_display_ids_normalized(
    experiment: ModuleType, name: str, expected: str
) -> None:
    driver = importlib.import_module("benchmark_rotation_nll_selection")
    assert driver.normalize_rms_display_name(name) == expected
