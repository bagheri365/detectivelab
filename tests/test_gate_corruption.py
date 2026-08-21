from __future__ import annotations

import pytest

from detectivelab.evaluation.gate_corruption import (
    _gold_for_item,
    apply_gate_corruption,
    select_corrupted_items,
)


def test_clean_gate_is_unchanged():
    assert apply_gate_corruption("present", "clean") == ("present", False)
    assert apply_gate_corruption("absent", "clean") == ("absent", False)


def test_false_absence_only_flips_present():
    assert apply_gate_corruption("present", "false_absence") == ("absent", True)
    assert apply_gate_corruption("absent", "false_absence") == ("absent", False)


def test_false_presence_only_flips_absent():
    assert apply_gate_corruption("absent", "false_presence") == ("present", True)
    assert apply_gate_corruption("present", "false_presence") == ("present", False)


def test_disabled_corruption_preserves_gate():
    assert apply_gate_corruption(
        "present", "false_absence", enabled=False
    ) == ("present", False)
    assert apply_gate_corruption(
        "absent", "false_presence", enabled=False
    ) == ("absent", False)


def test_partial_selection_is_deterministic_and_nested():
    ids = [f"scene_{i:04d}__conflict" for i in range(8)]
    q25 = select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=0.25
    )
    q50 = select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=0.50
    )
    q75 = select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=0.75
    )
    full = select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=1.00
    )

    assert len(q25) == 2
    assert len(q50) == 4
    assert len(q75) == 6
    assert len(full) == 8
    assert q25 <= q50 <= q75 <= full


def test_partial_selection_rounds_half_up_for_six_eligible_cases():
    ids = [f"item_{i}" for i in range(6)]
    assert len(select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=0.25
    )) == 2
    assert len(select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=0.50
    )) == 3
    assert len(select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=0.75
    )) == 5
    assert len(select_corrupted_items(
        ids, corruption="false_absence", corruption_rate=1.00
    )) == 6


def test_invalid_rate_rejected():
    with pytest.raises(ValueError):
        select_corrupted_items(
            ["a"], corruption="false_presence", corruption_rate=1.1
        )


def test_invalid_corruption_rejected():
    with pytest.raises(ValueError):
        apply_gate_corruption("present", "random")


def test_invalid_presence_rejected():
    with pytest.raises(ValueError):
        apply_gate_corruption("unknown", "clean")


def test_gold_is_read_from_questions_not_participant_payload(tmp_path):
    case_dir = tmp_path / "scene_0000"
    case_dir.mkdir()
    (case_dir / "payloads.json").write_text(
        '[{"scene_id":"scene_0000","item_id":"scene_0000__conflict","family":"conflict"}]',
        encoding="utf-8",
    )
    (case_dir / "questions.json").write_text(
        '[{"item_id":"scene_0000__conflict","family":"conflict","gold":"unknown"}]',
        encoding="utf-8",
    )

    assert _gold_for_item(case_dir, "scene_0000__conflict") == "unknown"


def test_gold_lookup_supports_nested_question_exports(tmp_path):
    case_dir = tmp_path / "scene_0001"
    case_dir.mkdir()
    (case_dir / "questions.json").write_text(
        '{"questions":[{"item_id":"scene_0001__conflict","answer":"contradicted"}]}',
        encoding="utf-8",
    )

    assert _gold_for_item(case_dir, "scene_0001__conflict") == "contradicted"
