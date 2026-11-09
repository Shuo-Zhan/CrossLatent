"""Compatibility alias for historical scripts; implementation lives in the package."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crosslatent._core import neuron_boosting as _implementation

sys.modules[__name__] = _implementation
globals().update({key: value for key, value in vars(_implementation).items() if not key.startswith("__")})
