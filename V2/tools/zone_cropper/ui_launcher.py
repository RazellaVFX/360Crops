from __future__ import annotations


def run_app(ui_choice: str = "qt") -> None:
    del ui_choice  # Portable build ships only the Qt interface.
    from .ui_qt import run_app as run_qt

    run_qt()
