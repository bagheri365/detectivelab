from __future__ import annotations

import pytest

from detectivelab.evaluation.abstaining_gate import (
    apply_abstention_protection,
    select_protected_items,
)


def test_unprotected_false_absence_remains_absent():
    state, abstained = apply_abstention_protection(
        clean_presence="present",
        false_absence_corrupted=True,
        protected=False,
    )
    assert state == "absent"
    assert abstained is False


def test_protected_false_absence_becomes_uncertain():
    state, abstained = apply_abstention_protection(
        clean_presence="present",
        false_absence_corrupted=True,
        protected=True,
    )
    assert state == "uncertain"
    assert abstained is True


def test_absent_case_is_not_changed_by_false_absence_protection():
    state, abstained = apply_abstention_protection(
        clean_presence="absent",
        false_absence_corrupted=False,
        protected=True,
    )
    assert state == "absent"
    assert abstained is False


def test_present_without_corruption_remains_present():
    state, abstained = apply_abstention_protection(
        clean_presence="present",
        false_absence_corrupted=False,
        protected=True,
    )
    assert state == "present"
    assert abstained is False


def test_protection_selection_is_nested_and_deterministic():
    ids = [f"scene_{i:04d}__conflict" for i in range(6)]

    p0 = select_protected_items(ids, 0.00)
    p25 = select_protected_items(ids, 0.25)
    p50 = select_protected_items(ids, 0.50)
    p75 = select_protected_items(ids, 0.75)
    p100 = select_protected_items(ids, 1.00)

    assert len(p0) == 0
    assert len(p25) == 2
    assert len(p50) == 3
    assert len(p75) == 5
    assert len(p100) == 6
    assert p0 <= p25 <= p50 <= p75 <= p100


def test_protection_selection_is_repeatable():
    ids = ["c", "a", "b", "d"]
    assert select_protected_items(ids, 0.50) == select_protected_items(
        reversed(ids), 0.50
    )


def test_invalid_protection_rate_rejected():
    with pytest.raises(ValueError):
        select_protected_items(["a"], -0.1)
    with pytest.raises(ValueError):
        select_protected_items(["a"], 1.1)


def test_invalid_presence_rejected():
    with pytest.raises(ValueError):
        apply_abstention_protection(
            clean_presence="unknown",
            false_absence_corrupted=True,
            protected=True,
        )
