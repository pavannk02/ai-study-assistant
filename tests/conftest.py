"""
conftest.py – pytest configuration for the ai-study-assistant test suite.

Adds the project root to sys.path so that 'from src.xxx import ...'
works in every test file without needing an editable install.
"""

import sys
from pathlib import Path

# Insert the project root (the folder containing src/) at the front of sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
