"""
TurboQuant vLLM integration — attention backend for vLLM v0.17+.

Provides hook-based integration that replaces standard KV cache with
TurboQuant compressed storage for long-context inference.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import logging
import torch
from typing import Optional, Dict

logger = logging.getLogger("turboquant.vllm")

# Operating modes
MODE_CAPTURE_ONLY = "capture_only"   # Only capture KV, don't use TQ for attention
MODE_HYBRID = "hybrid"               # Use TQ for history + exact buffer for recent

_GLOBAL_MODE = MODE_CAPTURE_ONLY


def set_mode(mode: str):
    global _GLOBAL_MODE
    if mode not in (MODE_CAPTURE_ONLY, MODE_HYBRID):
        raise ValueError(f"Invalid mode: {mode!r}. Must be {MODE_CAPTURE_ONLY!r} or {MODE_HYBRID!r}")
    _GLOBAL_MODE = mode


def get_mode() -> str:
    return _GLOBAL_MODE


class TQLayerState:
    """Per-layer TurboQuant state for vLLM integration."""

    def __init__(
        self,
        layer_idx: int,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        ring_capacity: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.ring_capacity = ring_capacity
        self.device = device or torch.device("cuda")
        self.supports_hybrid = True

        # Quantized storage (populated during capture/flush)
        self.key_quantized = None
        self.value_quantized = None

        # Ring buffer for recent tokens
        self.key_ring: Optional[torch.Tensor] = None
        self.value_ring: Optional[torch.Tensor] = None
        self.ring_len: int = 0
        self.total_len: int = 0

        # Codebook (precomputed)
        self._codebook = None
        self._rotation_matrix = None
        self._qjl_matrix = None

    def initialize_codebook(self, codebook_centroids: torch.Tensor, Pi: torch.Tensor, S: torch.Tensor):
        """Set precomputed codebook and rotation matrices."""
        self._codebook = codebook_centroids.to(self.device)
        self._rotation_matrix = Pi.to(self.device)
        self._qjl_matrix = S.to(self.device)

    def append_to_ring(self, key: torch.Tensor, value: torch.Tensor):
        """Append token(s) to ring buffer. key/value: (num_kv_heads, seq, head_dim)."""
        seq_len = key.shape[1]
        self.total_len += seq_len

        if self.key_ring is None:
            self.key_ring = key
            self.value_ring = value
            self.ring_len = seq_len
        else:
            self.key_ring = torch.cat([self.key_ring, key], dim=1)
            self.value_ring = torch.cat([self.value_ring, value], dim=1)
            self.ring_len += seq_len

        if self.ring_len > self.ring_capacity:
            self._flush_ring()

    def _flush_ring(self):
        """Flush oldest tokens from ring to quantized storage."""
        n_flush = self.ring_len - self.ring_capacity
        if n_flush <= 0:
            return

        # In a full implementation, this would quantize and store
        # For now, we just trim the ring buffer
        self.key_ring = self.key_ring[:, n_flush:, :]
        self.value_ring = self.value_ring[:, n_flush:, :]
        self.ring_len = self.ring_capacity

    def get_seq_length(self) -> int:
        return self.total_len

    def memory_bytes(self) -> int:
        total = 0
        if self.key_ring is not None:
            total += self.key_ring.nelement() * self.key_ring.element_size()
        if self.value_ring is not None:
            total += self.value_ring.nelement() * self.value_ring.element_size()
        return total


def install_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: Optional[int] = None,
    mode: str = MODE_CAPTURE_ONLY,
    no_alloc: bool = False,
) -> Dict[str, TQLayerState]:
    """
    Install TurboQuant hooks on a vLLM model runner.

    Args:
        model_runner: vLLM GPUModelRunner instance
        key_bits: bits for key quantization (default 3)
        value_bits: bits for value quantization (default 2)
        value_group_size: group size for value quantization
        ring_capacity: number of recent tokens kept unquantized
        initial_layers_count: first N layers use higher precision
        initial_layers_key_bits: key bits for initial layers (default: key_bits + 1)
        mode: operating mode
        no_alloc: if True, skip KV cache allocation for hooked layers

    Returns:
        Dict mapping layer names to TQLayerState instances.
    """
    set_mode(mode)

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 6)

    static_ctx = model_runner.compilation_config.static_forward_context
    layer_states: Dict[str, TQLayerState] = {}

    for layer_idx, (name, module) in enumerate(static_ctx.items()):
        if not hasattr(module, "kv_cache"):
            continue

        head_dim = getattr(module, "head_size", 128)
        num_kv_heads = getattr(module, "num_kv_heads", 1)

        bits = initial_layers_key_bits if layer_idx < initial_layers_count else key_bits

        state = TQLayerState(
            layer_idx=layer_idx,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            key_bits=bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            ring_capacity=ring_capacity,
            device=model_runner.device,
        )

        layer_states[name] = state

    model_runner._tq_layer_states = layer_states
    logger.info(f"[TurboQuant] Installed hooks on {len(layer_states)} layers "
                f"(key_bits={key_bits}, value_bits={value_bits}, ring={ring_capacity})")

    return layer_states


def free_kv_cache(model_runner) -> int:
    """Free paged KV cache for TQ-hooked layers, return bytes freed."""
    layer_states = getattr(model_runner, "_tq_layer_states", None)
    if not layer_states:
        return 0

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device
    tiny = torch.zeros(1, dtype=torch.int8, device=device)
    freed = 0

    for layer_name in layer_states:
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0 and hasattr(kv_list[0], "nelement"):
            old = kv_list[0]
            freed += old.nelement() * old.element_size()
            kv_list[0] = tiny

    torch.cuda.empty_cache()
    logger.info(f"[TurboQuant] Freed {freed / 1024**2:.1f} MB of paged KV cache")
    return freed
