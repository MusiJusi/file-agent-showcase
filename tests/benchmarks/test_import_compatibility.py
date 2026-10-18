"""Legacy imports must refer to the actual, patchable implementation modules."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("hf_batch", "hf.batch"),
        ("hf_cli", "hf.cli"),
        ("hf_dataset", "hf.dataset"),
        ("hf_output", "hf.output"),
        ("hf_rag", "hf.rag"),
        ("docbench_batch", "docbench.batch"),
        ("docbench_cli", "docbench.cli"),
        ("docbench_dataset", "docbench.dataset"),
        ("docbench_output", "docbench.output"),
    ],
)
def test_legacy_imports_preserve_module_identity_and_patching(legacy, canonical, monkeypatch):
    old_module = import_module(f"file_agent.{legacy}")
    new_module = import_module(f"file_agent.benchmarks.{canonical}")

    assert old_module is new_module
    marker = object()
    monkeypatch.setattr(f"file_agent.{legacy}.compatibility_probe", marker, raising=False)
    assert new_module.compatibility_probe is marker
