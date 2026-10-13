"""Future behavioral assertions plus active scaffold audits; TESTING strategy.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

from nta_zerodce.losses.registry import LOSS_CONTRACTS, LOSS_REGISTRY
from nta_zerodce.training.config import Config


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="M1-M6 implementation pending")
def test_contract_1() -> None:
    """load_config rejects unknown fields, negative/nonfinite weights, invalid crop/patch, curve/flow scale outside (0,1], and wrong scalar types.

    Args:
        No runtime arguments; construct the fixtures stated in the purpose.

    Returns:
        No value; assertions described above must pass when the stub is replaced.

    Method:
        docs/TESTING.md; remove xfail together with implementation, never to hide failures.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError(
        "TODO: load_config rejects unknown fields, negative/nonfinite weights, invalid crop/patch, curve/flow scale outside (0,1], and wrong scalar types"
    )


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="M1-M6 implementation pending")
def test_contract_2() -> None:
    """Defaults and full YAML carry every dataclass field; full denoise implies noise; adaptive disables baseline exp; each ablation row changes only flags and seed.

    Args:
        No runtime arguments; construct the fixtures stated in the purpose.

    Returns:
        No value; assertions described above must pass when the stub is replaced.

    Method:
        docs/TESTING.md; remove xfail together with implementation, never to hide failures.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError(
        "TODO: Defaults and full YAML carry every dataclass field; full denoise implies noise; adaptive disables baseline exp; each ablation row changes only flags and seed"
    )


# Active collection-time checks avoid implementing a function in this stub-only scaffold.
# make docs-check collects this module; broken links fail during collection.
ROOT = Path(__file__).resolve().parents[1]
for document in [
    ROOT / "README.md",
    ROOT.parent / "README.md",
    ROOT.parent / "web" / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]:
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
        target = target.split("#", 1)[0]
        if target and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            assert (document.parent / unquote(target.strip("<>"))).exists(), (document, target)

defaults = asdict(Config())
for config_file in ["baseline.yaml", "full.yaml"]:
    actual = yaml.safe_load((ROOT / "configs" / config_file).read_text(encoding="utf-8"))
    assert actual.keys() == defaults.keys()
    for section, fields in defaults.items():
        if isinstance(fields, dict):
            assert actual[section].keys() == fields.keys(), (config_file, section)
    assert not actual["flags"]["denoise"] or actual["flags"]["noise_aware"]
matrix = yaml.safe_load((ROOT / "configs" / "ablation_matrix.yaml").read_text(encoding="utf-8"))
assert len(matrix["runs"]) == 13
assert matrix["seeds"] == [17, 29, 43]
for row in matrix["runs"]:
    assert row["flags"].keys() == defaults["flags"].keys()
    assert not row["flags"]["denoise"] or row["flags"]["noise_aware"]

# Every handwritten callable must be exactly a docstring followed by a TODO raise.
for source in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]:
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert ast.get_docstring(node), (source, node.name)
            assert node.returns is not None, (source, node.name)
            assert all(a.annotation is not None for a in node.args.args if a.arg != "self")
            assert len(node.body) == 2 and isinstance(node.body[1], ast.Raise), (source, node.name)
            raised = node.body[1].exc
            assert isinstance(raised, ast.Call) and isinstance(raised.func, ast.Name)
            assert raised.func.id == "NotImplementedError"
            assert isinstance(raised.args[0], ast.Constant)
            assert str(raised.args[0].value).startswith("TODO: ")

assert LOSS_REGISTRY.keys() == LOSS_CONTRACTS.keys()
for loss_name, loss_spec in LOSS_REGISTRY.items():
    assert callable(loss_spec.callable)
    assert (loss_spec.flag, loss_spec.default_weight, loss_spec.required_inputs) == LOSS_CONTRACTS[
        loss_name
    ][1:]
