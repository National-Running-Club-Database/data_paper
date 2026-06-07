"""Optional tqdm progress for long-running analysis loops."""

from __future__ import annotations

import os
import sys
from typing import Iterable, TypeVar

T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
    disable: bool | None = None,
    **kwargs,
) -> Iterable[T]:
    """Wrap *iterable* with tqdm when stderr is a TTY. Set NRCD_NO_PROGRESS=1 to disable."""
    if disable is None:
        disable = os.environ.get("NRCD_NO_PROGRESS") == "1" or not sys.stderr.isatty()
    try:
        from tqdm import tqdm

        return tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            leave=leave,
            file=sys.stderr,
            disable=disable,
            **kwargs,
        )
    except ImportError:
        return iterable
