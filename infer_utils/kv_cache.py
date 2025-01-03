# -*- encoding: utf-8 -*-
import torch
from transformers.cache_utils import Cache
from typing import Any, Dict, Optional, Tuple


class DynamicCache(Cache):
    def __init__(self, config, max_length: int, device: int, dtype) -> None:
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.config = config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.key_cache = torch.empty(
            self.num_layers,
            self.max_length,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.value_cache = torch.empty(
            self.num_layers,
            self.max_length,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.reset()

    def reset(self):
        self.cur_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # key value shape: 1, kv_len, num_heads, head_dim
        # Update the number of seen tokens
        kv_len = key_states.shape[1]
        if layer_idx == 0:
            self.cur_len += kv_len
        if self.cur_len > self.max_length:
            raise ValueError("The cache has reached its maximum length")
        # Update the cache
        self.key_cache[layer_idx, self.cur_len - kv_len : self.cur_len, :, :] = (
            key_states[0]
        )
        self.value_cache[layer_idx, self.cur_len - kv_len : self.cur_len, :, :] = (
            value_states[0]
        )
        return (
            self.key_cache[layer_idx, : self.cur_len].unsqueeze(0),
            self.value_cache[layer_idx, : self.cur_len].unsqueeze(0),
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.cur_len
