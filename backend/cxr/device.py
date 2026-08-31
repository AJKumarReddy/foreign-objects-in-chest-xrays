"""Device selection.

The notebook hardcoded ``torch.device('cuda:0')``; this resolves the best
available device instead so the same code runs on a CPU laptop and a GPU box.
"""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Return a usable :class:`torch.device` for ``preference``.

    ``auto`` picks CUDA, then Apple MPS, then CPU. An explicit request for an
    unavailable device falls back to CPU rather than raising.
    """

    pref = (preference or "auto").strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return torch.device(pref)


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index or 0
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return str(device)
