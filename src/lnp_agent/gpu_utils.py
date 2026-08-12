"""GPU memory management utilities for v2.3."""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def clear_gpu_memory(reason: str = "") -> None:
    """Aggressively free GPU memory.

    Calls gc.collect() to release Python reference cycles, then
    torch.cuda.empty_cache() to release cached CUDA memory blocks
    back to the OS.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    if reason:
        logger.info(f"GPU memory cleared: {reason}")
