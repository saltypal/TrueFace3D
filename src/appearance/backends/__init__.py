"""One adapter per appearance method (common available()/run() interface)."""
import importlib

NAMES = ("raw_projection", "deca", "opendelight", "trust")


def load(name):
    return importlib.import_module(f"{__name__}.{name}")


def availability():
    return {name: load(name).available() for name in NAMES}
