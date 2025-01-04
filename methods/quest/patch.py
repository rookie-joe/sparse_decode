from methods.quest.attention import QuestFlashAttention
from infer_utils.modeling_qwen2 import Qwen2ForCausalLM, Qwen2FlashAttention2


def patch_model(
    model: Qwen2ForCausalLM,
    max_new_tokens: int,
    page_size: int,
    page_topk: int,
    verbose: bool = False,
):
    for name, module in model.named_modules():
        if isinstance(module, Qwen2FlashAttention2):
            module.flash_attention = QuestFlashAttention(
                max_new_tokens, page_size, page_topk, verbose
            )
    print("Replace flash attention with quest flash attention successfully")
