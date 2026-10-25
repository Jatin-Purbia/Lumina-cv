"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def _check_same_shape(a: Tensor, b: Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


def noise_loss(predicted_sigma: Tensor, target_sigma: Tensor) -> Tensor:
    """Supervise the per-pixel noise branch.

    Args:
        predicted_sigma: (B,1,H,W) in [0,0.2].
        target_sigma: Detached (B,1,H,W), synthetic unclipped standard deviation.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 1; L_noise = mean |sigma_hat - sigma_target| (target detached).
        See docs/METHOD.md; extensions are project proposals.

    Raises:
        ValueError: If the shapes differ.
    """
    _check_same_shape(predicted_sigma, target_sigma)
    return (predicted_sigma - target_sigma.detach()).abs().mean()


def invariance_loss(noisy_output: Tensor, teacher_output: Tensor) -> Tensor:
    """Enforce enhancement invariance to independent added noise.

    Args:
        noisy_output: RGB (B,3,H,W) in [0,1], retains graph.
        teacher_output: Detached RGB (B,3,H,W) in [0,1].

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 1; mean |E(x+n)-stopgrad(E(x))|. See docs/METHOD.md; extensions are project
        proposals.

    Raises:
        ValueError: If the shapes differ.
    """
    _check_same_shape(noisy_output, teacher_output)
    return (noisy_output - teacher_output.detach()).abs().mean()


def fidelity_loss(output: Tensor, teacher_output: Tensor) -> Tensor:
    """Anchor synthetic denoising to clean-input curve output.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        teacher_output: Detached clean-input enhanced_raw (B,3,H,W) in [0,1].

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 1; L1 synthetic fidelity mean |E(x+n).enhanced - stopgrad(E(x).enhanced_raw)|;
        requires denoise, uses no real high-light GT. See docs/METHOD.md; extensions are project
        proposals.

    Raises:
        ValueError: If the shapes differ.
    """
    _check_same_shape(output, teacher_output)
    return (output - teacher_output.detach()).abs().mean()
