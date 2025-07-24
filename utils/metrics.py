import torch


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        estimate: (B, T) estimated waveform
        reference: (B, T) reference waveform
    Returns:
        mean SI-SDR over the batch (scalar tensor)
    """
    if estimate.shape != reference.shape:
        raise ValueError("Mismatching shapes for SI-SDR computation")

    ref_energy = torch.sum(reference ** 2, dim=1, keepdim=True) + eps
    projection = torch.sum(estimate * reference, dim=1, keepdim=True) * reference / ref_energy
    noise = estimate - projection
    ratio = (torch.sum(projection ** 2, dim=1) + eps) / (torch.sum(noise ** 2, dim=1) + eps)
    return 10 * torch.log10(ratio).mean()


def psnr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (PSNR) in dB (peak = 1)."""
    mse = torch.mean((estimate - reference) ** 2, dim=1) + eps
    return 10 * torch.log10(1.0 / mse).mean() 