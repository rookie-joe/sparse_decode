import torch
import math
from flash_attn import flash_attn_func


class QuestFlashAttention:
    def __init__(
        self,
        max_new_tokens: int,
        page_size: int,
        page_topk: int,
        verbose=False,
    ):
        self.max_new_tokens = max_new_tokens
        self.page_size = page_size
        self.page_topk = page_topk
        self.verbose = verbose

    def __init_quest(self, key_states: torch.Tensor):
        batch_size, k_len, num_kv_heads, head_dim = key_states.shape
        assert batch_size == 1, "only support batch_size=1 for now"
        max_len = (
            math.ceil((k_len + self.max_new_tokens) / self.page_size) * self.page_size
        )
        page_num = max_len // self.page_size
        cur_page_num = k_len // self.page_size
        # init key value cache with limited size
        self.__key_cache = torch.zeros(
            1,
            (self.page_topk + 1) * self.page_size,
            num_kv_heads,
            head_dim,
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.__value_cache = torch.zeros(
            1,
            (self.page_topk + 1) * self.page_size,
            num_kv_heads,
            head_dim,
            dtype=key_states.dtype,
            device=key_states.device,
        )
        # init page indexs
        self.page_idxs = torch.arange(
            max_len, dtype=torch.int32, device=key_states.device
        ).view(page_num, self.page_size)
        # init min key and max key
        self.__key_min = torch.zeros(
            1,
            page_num,
            num_kv_heads,
            head_dim,
            dtype=key_states.dtype,
            device=key_states.device,
        ).fill_(-torch.inf)
        self.__key_max = torch.zeros(
            1,
            page_num,
            num_kv_heads,
            head_dim,
            dtype=key_states.dtype,
            device=key_states.device,
        ).fill_(-torch.inf)
        # fill key_min and key_max
        self.__key_min[:, :cur_page_num, :, :] = (
            key_states[:, : cur_page_num * self.page_size, :, :]
            .view(1, cur_page_num, self.page_size, num_kv_heads, head_dim)
            .min(dim=2)
            .values
        )
        self.__key_max[:, :cur_page_num, :, :] = (
            key_states[:, : cur_page_num * self.page_size, :, :]
            .view(1, cur_page_num, self.page_size, num_kv_heads, head_dim)
            .max(dim=2)
            .values
        )

    def __update_quest(self, key_states: torch.Tensor):
        k_len = key_states.shape[1]
        # get page id
        page_id = (k_len - 1) // self.page_size
        in_page_id = (k_len - 1) % self.page_size
        # the last page is not full now, no need to update
        if in_page_id != self.page_size - 1:
            return
        # update key_min and key_max
        self.__key_min[:, page_id, :, :] = (
            key_states[:, -self.page_size :, :, :].min(dim=1, keepdim=True).values
        )
        self.__key_max[:, page_id, :, :] = (
            key_states[:, -self.page_size :, :, :].max(dim=1, keepdim=True).values
        )

    def __retrieve_kv(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        batch_size, q_len, num_heads, head_dim = query_states.shape
        batch_size, k_len, num_kv_heads, head_dim = key_states.shape
        num_shared_q_head = num_heads // num_kv_heads
        # prefill phase or current kv is small enough, don't need to select topk
        if q_len > 1 or k_len <= self.page_size * (self.page_topk + 1):
            return key_states, value_states
        # get page id
        page_id = (k_len - 1) // self.page_size
        in_page_id = (k_len - 1) % self.page_size
        final_kv_len = min(self.page_size * self.page_topk + in_page_id + 1, k_len)
        # quest score
        # [batch_size, 1, num_q_head, head_dim] x [batch_size, page_num, num_kv_head, head_dim].repeat -> [batch_size, page_num, num_q_head, head_dim]
        # [batch_size, page_num, num_q_head, head_dim] -> [batch_size, page_num, num_q_head]
        score = torch.sum(
            torch.max(
                query_states
                * self.__key_min[:batch_size, :page_id, ...].repeat_interleave(
                    num_shared_q_head, 2
                ),
                query_states
                * self.__key_max[:batch_size, :page_id, ...].repeat_interleave(
                    num_shared_q_head, 2
                ),
            ),
            dim=-1,
        )
        # select by kv heads, average the score for shared query heads for each kv head
        # shape become [batch_size, page_num, num_kv_head]
        score = score.view(
            score.shape[0], score.shape[1], num_kv_heads, num_shared_q_head
        )
        score = score.mean(dim=-1)
        # set score of block 1 = inf
        score[:, 0, :] = float("inf")
        # select topk page and last page, [batch_size, page_topk+1, num_kv_head]
        topk_page = torch.sort(
            torch.topk(score, min(page_id, self.page_topk), dim=1).indices, dim=1
        ).values
        topk_page = torch.cat(
            (
                topk_page,
                torch.zeros(
                    batch_size,
                    1,
                    num_kv_heads,
                    device=topk_page.device,
                    dtype=topk_page.dtype,
                )
                + page_id,
            ),
            dim=1,
        )
        # FIXME: for loop maybe too slow
        for h in range(num_kv_heads):
            self.__key_cache[:, :final_kv_len, h, :] = key_states[
                :, self.page_idxs[topk_page[0, :, h]].view(-1)[:final_kv_len], h, :
            ]
            self.__value_cache[:, :final_kv_len, h, :] = value_states[
                :, self.page_idxs[topk_page[0, :, h]].view(-1)[:final_kv_len], h, :
            ]
        return (
            self.__key_cache[:, :final_kv_len, :, :],
            self.__value_cache[:, :final_kv_len, :, :],
        )

    def __call__(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        **kwargs
    ):
        batch_size, q_len, num_heads, head_dim = query_states.shape
        if q_len > 1:  # prefill
            self.__init_quest(key_states)
        else:  # decode
            self.__update_quest(key_states)
        key_states, value_states = self.__retrieve_kv(
            query_states, key_states, value_states
        )
        if self.verbose:
            print(
                "\r",
                "key value shape:",
                key_states.shape,
                value_states.shape,
                end="",
                flush=True,
            )
        return flash_attn_func(query_states, key_states, value_states, causal=q_len > 1)
