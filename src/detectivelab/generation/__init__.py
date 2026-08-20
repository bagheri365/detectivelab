"""Synthetic benchmark generation."""

from .questions import BenchmarkItem, generate_questions
from .scenes import generate_scene

__all__ = ["BenchmarkItem", "generate_questions", "generate_scene"]
