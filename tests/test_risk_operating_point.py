from __future__ import annotations

from PIL import Image

from detectivelab.evaluation.risk_operating_point import (
    POLICIES,
    _quality_scores,
    policy_escalates,
)


def test_policy_set_contains_required_baselines():
    assert "NEVER_ESCALATE" in POLICIES
    assert "STABILITY_ONLY" in POLICIES
    assert "ANY_SIGNAL" in POLICIES
    assert "ALWAYS_ESCALATE" in POLICIES


def test_quality_scores_flat_image_has_zero_contrast_and_edges():
    image = Image.new("RGB", (16, 16), "white")
    contrast, edge = _quality_scores(image)
    assert contrast == 0.0
    assert edge == 0.0


def test_any_signal_policy():
    signals = {
        "instability": False,
        "low_contrast": True,
        "low_edge": False,
    }
    assert policy_escalates("ANY_SIGNAL", signals)


def test_two_plus_requires_two_signals():
    one = {
        "instability": True,
        "low_contrast": False,
        "low_edge": False,
    }
    two = {
        "instability": True,
        "low_contrast": True,
        "low_edge": False,
    }
    assert not policy_escalates("TWO_PLUS", one)
    assert policy_escalates("TWO_PLUS", two)


def test_never_and_always_baselines():
    signals = {
        "instability": False,
        "low_contrast": False,
        "low_edge": False,
    }
    assert not policy_escalates("NEVER_ESCALATE", signals)
    assert policy_escalates("ALWAYS_ESCALATE", signals)
