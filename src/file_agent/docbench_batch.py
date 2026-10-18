"""Compatibility import for :mod:`file_agent.benchmarks.docbench.batch`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("file_agent.benchmarks.docbench.batch")
