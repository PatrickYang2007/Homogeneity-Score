"""Shared pytest setup.

Puts src/ on sys.path so tests can `import model`, `import eval_report`, etc.,
exactly as the scripts do when run from src/. Also provides small synthetic
fixtures so no real genome / bedgraph data is ever needed (and never ends up in
the public repo).
"""
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))


@pytest.fixture
def toy_chrom():
    """A 200 bp deterministic 'chromosome' of A/C/G/T for slicing checks."""
    bases = "ACGT"
    return "".join(bases[i % 4] for i in range(200))
