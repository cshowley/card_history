"""
Shared pytest fixtures for data integrity metric tests.

Every test gets a fresh tracker via the ``fresh_tracker`` fixture, which
calls ``reset_tracker()`` before and after the test. This ensures no
cross-test pollution of the global singleton.
"""

import sys
import os

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_integrity import get_tracker, reset_tracker


def find_widget(tracker, widget_id):
    """Find a widget by ID in the tracker's widget list."""
    for w in tracker.get_data()["widgets"]:
        if w.get("id") == widget_id:
            return w
    return None


def find_all_widgets(tracker, widget_id):
    """Find all widgets matching an ID (useful for tables/charts)."""
    return [w for w in tracker.get_data()["widgets"] if w.get("id") == widget_id]


@pytest.fixture(autouse=True)
def fresh_tracker():
    """Reset the global tracker before and after every test."""
    reset_tracker()
    tracker = get_tracker()
    yield tracker
    reset_tracker()
