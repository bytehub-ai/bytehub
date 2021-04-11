import importlib
import os


available_backends = {}


def _import_backend(name):
    try:
        mod = importlib.import_module("." + name, __name__)
        available_backends[name] = mod.Store
    except ImportError:
        pass


for module in os.listdir(os.path.dirname(__file__)):
    if module.startswith("_") or module[-3:] != ".py":
        continue
    _import_backend(module[:-3])
del module
