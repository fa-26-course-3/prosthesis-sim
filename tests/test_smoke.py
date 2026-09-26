"""Pytest wrapper around scripts/smoke_test.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_end_to_end():
    from smoke_test import main
    main()  # will assert internally
