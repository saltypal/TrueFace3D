"""Load tensors from a Lightning/pickle checkpoint without executing its pickled objects.

Classes outside torch / collections / numpy are replaced by inert stubs, so a checkpoint's
hyper-parameter objects (omegaconf, custom configs) are never imported or constructed.
"""
from __future__ import annotations

import pickle
import types

import torch

_ALLOWED = ("torch", "collections", "numpy", "_codecs", "builtins")


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def __call__(self, *args, **kwargs):
        return _Stub()


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _ALLOWED and not (module == "builtins" and name in ("eval", "exec", "getattr", "__import__", "open")):
            return super().find_class(module, name)
        return _Stub


safe_pickle = types.ModuleType("safe_pickle")
safe_pickle.Unpickler = _Unpickler
safe_pickle.load = lambda f, **kw: _Unpickler(f, **kw).load()
safe_pickle.__name__ = "pickle"


def load_state_dict(path, key="state_dict") -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=safe_pickle)
    sd = ckpt.get(key, ckpt) if isinstance(ckpt, dict) else ckpt
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
