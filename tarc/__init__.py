"""TARC: Text-Anchored Reliability-Aware Correction.

Imports are lazy so command help and the saved-output demo need no model runtime.
"""

__version__ = "1.0.0"
__all__ = ["TARC", "TARCTextEncoder", "ModelConfig", "load_model", "predict_npz"]


def __getattr__(name):
    if name == "TARC":
        from .model import TARC

        return TARC
    if name == "TARCTextEncoder":
        from .bert_model import TARCTextEncoder

        return TARCTextEncoder
    if name == "ModelConfig":
        from .config import ModelConfig

        return ModelConfig
    if name in {"load_model", "predict_npz"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
