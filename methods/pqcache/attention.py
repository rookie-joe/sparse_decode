import time

import numpy
import numpy as np
import torch
from cuml.cluster import KMeans
from flash_attn import flash_attn_func


class PQCacheFlashAttention:
    def __init__(
        self,
        max_new_tokens: int,
        pq_k: int,  # k in product quantization
        pq_m: int,  # m in product quantization
        top_k: int = 32,
        local_window: int = 128,
        global_window: int = 64,
        verbose: bool = False,
    ):
        self.max_new_tokens = max_new_tokens
        self.num_clusters = pq_k
        self.num_subvectors = pq_m
        self.topk = top_k
        self.local_window = local_window
        self.global_window = global_window
        self.verbose = verbose

        # Initialize storage for PQ components
        self.codebooks = []
        self.codes = []
        self.cur_len = 0

        # For timing analysis
        self.timing = {"quantization": 0, "search": 0, "total": 0}

    def _init_pq(self, key_states: torch.Tensor):
        """Initialize Product Quantization for the initial key states"""
        batch_size, k_len, num_kv_heads, head_dim = key_states.shape
        assert batch_size == 1, "only support batch_size=1 for now"

        # Initialize storage for each head
        for kv_head_i in range(num_kv_heads):
            # Reshape key states for current head

            head_keys = (
                key_states[0, :, kv_head_i, :].cpu().float().numpy().astype(np.float32)
            )

            # Split dimensions into subvectors
            d_sub = head_dim // self.num_subvectors
            subvectors = [
                head_keys[:, i * d_sub : (i + 1) * d_sub]
                for i in range(self.num_subvectors)
            ]

            # Train kmeans for each subvector
            head_codebooks = []
            head_codes = np.zeros((k_len, self.num_subvectors), dtype=np.int32)

            for i, subvec in enumerate(subvectors):
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
                kmeans.fit(subvec)
                head_codebooks.append(kmeans)
                head_codes[:, i] = kmeans.predict(subvec)

            self.codebooks.append(head_codebooks)
            self.codes.append(head_codes)

        # Initialize key/value cache
        self.cur_len = k_len

    def _update_pq(self, key_states: torch.Tensor):
        """Update Product Quantization with new key states.

        The update follows these rules:
        1. New tokens are first kept as local tokens in GPU memory
        2. Only when tokens are evicted from the local window, they are quantized
           and added to the PQ cache
        3. Bulk quantization happens when we have accumulated enough tokens to
           be evicted from local window

        Args:
            key_states: shape [batch_size, k_len, num_kv_heads, head_dim]
        """
        batch_size, k_len, num_kv_heads, head_dim = key_states.shape
        assert batch_size == 1, "only support batch_size=1 for now"

        # Calculate how many tokens we have that haven't been PQ encoded yet
        tokens_not_in_pq = self.cur_len - self.codes[0].shape[0]

        # If we have accumulated more tokens than local window size,
        # we need to quantize the excess tokens
        if tokens_not_in_pq >= self.local_window:
            # Start from where we last encoded
            start_idx = self.codes[0].shape[0]

            # End at current tokens minus local window
            end_idx = start_idx + tokens_not_in_pq

            # Quantize tokens for each head
            d_sub = head_dim // self.num_subvectors
            for kv_head_i in range(num_kv_heads):
                # Get the tokens that need to be quantized from key_cache
                tokens_to_quantize = (
                    key_states[0, start_idx:end_idx, kv_head_i]
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )
                new_codes = np.zeros(
                    (tokens_not_in_pq, self.num_subvectors), dtype=np.int32
                )

                # Process all subvectors for this batch of tokens
                for i in range(self.num_subvectors):
                    subvecs = tokens_to_quantize[:, i * d_sub : (i + 1) * d_sub]
                    kmeans = self.codebooks[kv_head_i][i]

                    new_codes[:, i] = kmeans.predict(subvecs)

                # Append new codes
                self.codes[kv_head_i] = np.concatenate(
                    [self.codes[kv_head_i], new_codes]
                )

    def _pq_search(
        self, query: torch.Tensor, kv_head_idx: int, local_indice: numpy.ndarray
    ) -> np.ndarray:
        """Search for top-k nearest neighbors using PQ distances"""
        num_groups = query.shape[0]
        d_sub = query.shape[-1] // self.num_subvectors
        query_np = query.cpu().float().numpy().astype(np.float32)

        # Initialize distances array for all queries in the group
        distances = np.zeros(
            (self.codes[kv_head_idx].shape[0], self.num_subvectors, num_groups)
        )

        # Process each subvector
        for i in range(self.num_subvectors):
            centroids = self.codebooks[kv_head_idx][i].cluster_centers_
            query_subs = query_np[:, i * d_sub : (i + 1) * d_sub]

            centroid_distances = np.linalg.norm(
                centroids[:, None, :] - query_subs[None, :, :], axis=2
            )

            distances[:, i, :] = centroid_distances[self.codes[kv_head_idx][:, i]]

        total_distances = distances.sum(axis=1)
        query_contributions = total_distances.mean(axis=1)

        # Create a mask for indices that are not in local_indice
        mask = ~np.isin(np.arange(len(query_contributions)), local_indice)

        # Apply mask to get valid indices and their corresponding distances
        valid_indices = np.arange(len(query_contributions))[mask]
        valid_distances = query_contributions[mask]

        # Get top-k from valid indices only
        topk_idx = np.argsort(valid_distances)[: self.topk]
        topk_indices = valid_indices[topk_idx]

        return topk_indices

    def _get_local_tokens(self) -> np.ndarray:
        """Get indices for both local and global attention windows.

        Returns indices in two parts:
        1. Global window: tokens from the beginning of sequence
        2. Local window: most recent tokens

        Returns:
            np.ndarray: Combined unique indices from both windows
        """
        # Get recent context (local window)
        local_start = max(0, self.cur_len - self.local_window)
        local_indices = np.arange(local_start, self.cur_len)

        # Get early context (global window)
        global_end = min(self.global_window, self.cur_len)
        global_indices = np.arange(0, global_end)

        # Combine and deduplicate indices
        combined_indices = np.unique(np.concatenate([global_indices, local_indices]))

        return combined_indices

    def __call__(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        **kwargs
    ):
        start_time = time.time()

        batch_size, q_len, num_heads, head_dim = query_states.shape
        _, _, num_kv_heads, _ = key_states.shape
        num_groups = num_heads // num_kv_heads

        if q_len > 1:  # prefill phase
            self._init_pq(key_states)
            output = flash_attn_func(
                query_states, key_states, value_states, causal=True
            )

        else:  # decode phase
            self._update_pq(key_states)

            # For each kv head, find relevant tokens
            selected_indices = []
            for kv_head_i in range(num_kv_heads):

                # Get local context window
                local_indices = self._get_local_tokens()

                # Get top-k tokens based on PQ distance
                topk_indices = self._pq_search(
                    query_states[
                        0, 0, kv_head_i * num_groups : (kv_head_i + 1) * num_groups
                    ],
                    kv_head_i,
                    local_indices,
                )

                # Combine and deduplicate indices
                head_indices = np.unique(np.concatenate([topk_indices, local_indices]))

                selected_indices.append(head_indices)

            # Initialize reduced states with negative infinity for proper masking
            reduced_key = torch.zeros(
                (batch_size, len(selected_indices[0]), num_kv_heads, head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            reduced_value = torch.zeros_like(reduced_key)

            # Fill reduced states
            for kv_head_i, indices in enumerate(selected_indices):
                reduced_key[0, : len(indices), kv_head_i, ...] = key_states[
                    0, indices, kv_head_i, ...
                ]
                reduced_value[0, : len(indices), kv_head_i, ...] = value_states[
                    0, indices, kv_head_i, ...
                ]

            # Repeat for grouped attention if needed
            if num_groups > 1:
                reduced_key = reduced_key.repeat_interleave(num_groups, dim=2)
                reduced_value = reduced_value.repeat_interleave(num_groups, dim=2)

            output = flash_attn_func(
                query_states, reduced_key, reduced_value, causal=False
            )

        self.cur_len += 1

        # # Update cache
        # if q_len == 1:
        #     self.key_cache = torch.cat([
        #         self.key_cache,
        #         key_states
        #     ], dim=1)
        #     self.value_cache = torch.cat([
        #         self.value_cache,
        #         value_states
        #     ], dim=1)
        #     self.cur_len += 1
        #
        self.timing["total"] += time.time() - start_time
        #
        # if self.verbose:
        #     print(f"\rCurrent length: {self.cur_len}, "
        #           f"Selected tokens: {max_indices if q_len == 1 else self.cur_len}",
        #           end="", flush=True)

        return output
