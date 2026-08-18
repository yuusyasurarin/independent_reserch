"""研究用スクリプトをプロジェクト内venvで確実に実行するための共通処理。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"


def ensure_project_venv() -> None:
    """別のPythonで起動された場合、研究用venvのPythonで同じコマンドを再実行する。"""
    current = os.path.normcase(os.path.abspath(sys.executable))
    expected = os.path.normcase(os.path.abspath(VENV_PYTHON))
    if current == expected:
        return

    if not VENV_PYTHON.is_file():
        raise RuntimeError(
            f"研究用仮想環境が見つかりません: {VENV_PYTHON}\n"
            "READMEの手順に従って .venv を作成してください。"
        )

    os.environ["VIRTUAL_ENV"] = str(VENV_DIR)
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])
