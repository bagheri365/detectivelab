from __future__ import annotations

import pytest

from detectivelab.evaluation.evidence_uncertainty import (
    CALIBRATED_VIEWS,
    classify_presence_votes,
)


def test_calibrated_views_exclude_brightness():
    assert not any(view.startswith("brightness_") for view in CALIBRATED_VIEWS)


def test_calibrated_views_match_audited_set():
    assert CALIBRATED_VIEWS == (
        "original",
        "blur_020",
        "blur_040",
        "blur_060",
        "downsample_090",
        "downsample_075",
        "downsample_060",
    )


def test_unanimous_present_is_hard_present():
    state, agreement = classify_presence_votes(["present"] * 7)
    assert state == "present"
    assert agreement == 1.0


def test_unanimous_absent_is_hard_absent():
    state, agreement = classify_presence_votes(["absent"] * 7)
    assert state == "absent"
    assert agreement == 1.0


def test_single_disagreement_is_uncertain():
    state, agreement = classify_presence_votes(
        ["present", "present", "present", "present", "present", "present", "absent"]
    )
    assert state == "uncertain"
    assert agreement == pytest.approx(6 / 7)


def test_two_disagreements_are_uncertain():
    state, agreement = classify_presence_votes(
        ["present", "present", "present", "present", "present", "absent", "absent"]
    )
    assert state == "uncertain"
    assert agreement == pytest.approx(5 / 7)


def test_empty_votes_rejected():
    with pytest.raises(ValueError):
        classify_presence_votes([])


def test_invalid_vote_rejected():
    with pytest.raises(ValueError):
        classify_presence_votes(["present", "unknown"])
