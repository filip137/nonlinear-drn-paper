import torch


def amp_value(value):
    """Return the tensor/float value for an amplification source."""
    if hasattr(value, "state"):
        return value.state
    return value


def amp_detached_float(value) -> float:
    """Return an amplification source as a Python float for logging/control flow."""
    value = amp_value(value)
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def amp_ratio(current_amp, voltage_amp, exponent: int = 1):
    """Return (current_amp / voltage_amp) ** exponent, preserving tensors."""
    exponent = int(exponent)
    if exponent == 0:
        return 1.0
    return (amp_value(current_amp) / amp_value(voltage_amp)) ** exponent


def tensor_like(value, like: torch.Tensor):
    """Convert a scalar or tensor to the dtype/device of another tensor."""
    if torch.is_tensor(value):
        return value.to(dtype=like.dtype, device=like.device)
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)
