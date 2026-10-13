"""Future behavioral assertions plus active scaffold audits; TESTING strategy.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="M1-M6 implementation pending")
def test_contract_1() -> None:
    """Baseline forward on B=2,H=64,W=80 returns RGB enhanced/raw, A=(2,24,64,80), zero sigma=(2,1,64,80); finite ranges and exact device match.

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
        "TODO: Baseline forward on B=2,H=64,W=80 returns RGB enhanced/raw, A=(2,24,64,80), zero sigma=(2,1,64,80); finite ranges and exact device match"
    )


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="M1-M6 implementation pending")
def test_contract_2() -> None:
    """Noise+denoise forward accepts RGB only, internally concatenates sigma; gradients reach all enabled heads; light keeps identical output shapes.

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
        "TODO: Noise+denoise forward accepts RGB only, internally concatenates sigma; gradients reach all enabled heads; light keeps identical output shapes"
    )


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="M1-M6 implementation pending")
def test_contract_3() -> None:
    """curve_scale=.5 returns original resolution maps and outputs; baseline enhanced equals raw; A_override applies supplied maps.

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
        "TODO: curve_scale=.5 returns original resolution maps and outputs; baseline enhanced equals raw; A_override applies supplied maps"
    )
