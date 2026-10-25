"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.losses._ops import luminance, pool_mean


def spatial_loss(image: Tensor, output: Tensor, patch_size: int = 4) -> Tensor:
    """Preserve local intensity contrast.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        patch_size: Non-overlapping region size, default 4.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Baseline L_spa; valid four-neighbour region pairs, squared absolute differences.
        With u=P_k(Y(x)), v=P_k(Y(o)) and K=B*pooled_H*pooled_W the loss is
        (1/K) * sum over regions i and their valid neighbours j of (|v_i-v_j|-|u_i-u_j|)^2.
        Every unordered neighbour pair is therefore visited from both ends (factor two, the
        literal reading of the METHOD sum); boundary regions have fewer neighbours and no padded
        pairs are penalized. See docs/METHOD.md; baseline attribution [R1,R2].

    Raises:
        ValueError: If the inputs are not RGB tensors of equal shape or are smaller than a patch.
    """
    if image.shape != output.shape:
        raise ValueError(f"image {tuple(image.shape)} and output {tuple(output.shape)} differ")
    u = pool_mean(luminance(image), patch_size)
    v = pool_mean(luminance(output), patch_size)
    count = v.shape[0] * v.shape[2] * v.shape[3]
    # Horizontal and vertical neighbour pairs; each is visited from both of its regions.
    h_diff = (v[..., :, 1:] - v[..., :, :-1]).abs() - (u[..., :, 1:] - u[..., :, :-1]).abs()
    v_diff = (v[..., 1:, :] - v[..., :-1, :]).abs() - (u[..., 1:, :] - u[..., :-1, :]).abs()
    total = (h_diff * h_diff).sum() + (v_diff * v_diff).sum()
    return 2.0 * total / count
