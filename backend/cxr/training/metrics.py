"""Lightweight training meters.

Trimmed from the torchvision reference utilities the notebook vendored: the
distributed all-gather/pickle machinery is gone (this project trains on a single
device) and what remains is the running-average bookkeeping plus a progress log.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from collections.abc import Iterable

import torch

from cxr.logging_conf import get_logger

logger = get_logger(__name__)


class SmoothedValue:
    """Tracks a series of values with a windowed median and a global average."""

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        return float(torch.tensor(list(self.deque)).median().item()) if self.deque else 0.0

    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        return float(torch.tensor(list(self.deque), dtype=torch.float32).mean().item())

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Named :class:`SmoothedValue` meters plus an iteration progress logger."""

    def __init__(self, delimiter: str = "  ") -> None:
        self.meters: dict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, (float, int)):
                raise TypeError(f"meter {key!r} got a {type(value).__name__}")
            self.meters[key].update(float(value))

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def __getattr__(self, attr: str) -> SmoothedValue:
        meters = self.__dict__.get("meters", {})
        if attr in meters:
            return meters[attr]
        raise AttributeError(attr)

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def log_every(self, iterable: Iterable, print_freq: int, header: str = ""):
        i = 0
        total = len(iterable)  # type: ignore[arg-type]
        start = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")

        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if print_freq and (i % print_freq == 0 or i == total - 1):
                eta = datetime.timedelta(seconds=int(iter_time.global_avg * (total - i)))
                message = (
                    f"{header} [{i:>{len(str(total))}}/{total}] "
                    f"eta: {eta}  {self}  time: {iter_time}"
                )
                if torch.cuda.is_available():
                    message += f"  mem: {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB"
                logger.info(message)
            i += 1
            end = time.time()

        elapsed = time.time() - start
        logger.info(
            "%s total time: %s (%.4f s/it)",
            header,
            datetime.timedelta(seconds=int(elapsed)),
            elapsed / max(total, 1),
        )
