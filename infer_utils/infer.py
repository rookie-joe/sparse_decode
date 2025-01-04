# -*- encoding: utf-8 -*-
import time
from typing import List, Optional

import torch
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from infer_utils.kv_cache import DynamicCache


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def norm_logits(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: float,
    top_p: float,
) -> torch.Tensor:
    if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
        logits_shape = logits.shape
        logits = top_k_top_p_filtering(
            logits.view(-1, logits.shape[-1]) / temperature, top_k, top_p
        )
        probs = torch.nn.functional.softmax(logits.view(logits_shape), dim=-1)
    else:
        probs = logits.softmax(-1)
    return probs


def max_fn(x):
    # norm(max(x, 0))
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def top_k_top_p_sample(
    logits: torch.Tensor,
    top_k: float,
    top_p: float,
    do_sample: bool,
    temperature: float,
    num_samples: int = 1,
):
    if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
        probs = norm_logits(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        output_ids = torch.multinomial(probs[:, -1, :], num_samples=num_samples)
    else:
        if num_samples == 1:
            output_ids = logits[:, -1, :].argmax(-1, keepdim=True)
        else:
            output_ids = logits[:, -1, :].topk(num_samples, dim=-1).indices
    return output_ids


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    # cache kwargs
    kv_cache: DynamicCache = None,
    max_length: int = 8192,
    # generate kwargs
    max_new_tokens: int = 64,
    do_sample: bool = False,
    top_k: int = 0,
    top_p: float = 0.7,
    temperature: float = 1,
    **kwargs,
):
    assert input_ids.size(0) == 1, "only support batch size 1 for now"
    current_input_ids = input_ids
    generate_ids = torch.empty(
        input_ids.size(0), max_new_tokens, dtype=torch.long, device=model.device
    )
    if not isinstance(kv_cache, DynamicCache) or kv_cache.max_length < max_length:
        kv_cache = DynamicCache(model.config, max_length, model.device, model.dtype)
    # reset cache
    kv_cache.reset()
    # start generate
    for step in range(max_new_tokens):
        # forward pass, get logits
        output = model(
            input_ids=current_input_ids,
            past_key_values=kv_cache,
            return_dict=True,
            use_cache=True,
            **kwargs,
        )
        # only need logits for last token
        logits = output["logits"][:, -1:]
        # sample output id from logits
        output_ids = top_k_top_p_sample(
            logits,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        generate_ids[:, step] = output_ids
        # prepare for next generation step
        current_input_ids = output_ids
        # update kv cache
        kv_cache = output.past_key_values
        # stop generation if meet eos token
        if current_input_ids.item() == tokenizer.eos_token_id:
            break

    # return final generate ids
    step = min(step + 1, max_new_tokens)
    generate_ids = generate_ids[:, :step]
    return {"generate_ids": generate_ids, "kv_cache": kv_cache}


@torch.no_grad()
def stream_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    # cache kwargs
    kv_cache: DynamicCache = None,
    max_length: int = 8192,
    # generate kwargs
    max_new_tokens: int = 64,
    do_sample: bool = False,
    top_k: int = 0,
    top_p: float = 0.7,
    temperature: float = 1,
    **kwargs,
):
    assert input_ids.size(0) == 1, "only support batch size 1 for now"
    current_input_ids = input_ids
    if not isinstance(kv_cache, DynamicCache) or kv_cache.max_length < max_length:
        kv_cache = DynamicCache(model.config, max_length, model.device, model.dtype)
    # reset cache
    kv_cache.reset()
    # start generate
    for step in range(max_new_tokens):
        # forward pass, get logits
        output = model(
            input_ids=current_input_ids,
            past_key_values=kv_cache,
            return_dict=True,
            use_cache=True,
            **kwargs,
        )
        # only need logits for last token
        logits = output["logits"][:, -1:]
        # sample output id from logits
        output_ids = top_k_top_p_sample(
            logits,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        # prepare for next generation step
        current_input_ids = output_ids
        # update kv cache
        kv_cache = output.past_key_values
        # yield the generated output_ids
        yield {"generate_ids": output_ids, "kv_cache": kv_cache}
        # stop generation if meet eos token
        if output_ids.item() == tokenizer.eos_token_id:
            break


def infer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    # cache kwargs
    kv_cache: DynamicCache = None,
    max_length: int = 8192,
    # stream kwargs
    stream: bool = False,
    verbose: bool = False,
    # generate kwargs
    max_new_tokens: int = 64,
    do_sample: bool = False,
    top_k: int = 0,
    top_p: float = 0.7,
    temperature: float = 1,
    **kwargs,
):
    # encode prompt
    try:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

    except (TypeError, ValueError) as e:
        print(f"catch error ：{e}")
        input_ids = tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False
        ).input_ids.to(model.device)

    if not stream:
        # generate new token ids
        generate_dict = generate(
            model,
            tokenizer,
            input_ids,
            kv_cache,
            max_length,
            max_new_tokens,
            do_sample,
            top_k,
            top_p,
            temperature,
            **kwargs,
        )
        generate_ids = generate_dict["generate_ids"]
        # decode token ids to text
        output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        if verbose:
            print(output)
        output_len = generate_ids.shape[-1]
        kv_cache = generate_dict["kv_cache"]
    else:
        output_len = 0
        output = ""
        for generate_dict in stream_generate(
            model,
            tokenizer,
            input_ids,
            kv_cache,
            max_length,
            max_new_tokens,
            do_sample,
            top_k,
            top_p,
            temperature,
            **kwargs,
        ):
            generate_ids = generate_dict["generate_ids"]
            decoded_token = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            output += decoded_token
            if verbose:
                print(decoded_token, end="", flush=True)
            output_len += generate_ids.shape[-1]
            kv_cache = generate_dict["kv_cache"]
    # return final results
    results = {}
    results["prompt"] = prompt
    results["input_len"] = input_ids.shape[-1]
    results["output"] = output
    results["output_len"] = output_len
    results["kv_cache"] = kv_cache
    return results


def batch_infer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: List[str],
    system_prompt: Optional[str] = None,
    # cache kwargs
    kv_cache: DynamicCache = None,
    max_length: int = 8192,
    # generate kwargs
    max_new_tokens: int = 64,
    do_sample: bool = False,
    top_k: int = 0,
    top_p: float = 0.7,
    temperature: float = 1,
    # tqdm kwargs
    verbose: bool = True,
    **kwargs,
):
    kv_cache = None
    tokens_per_sec = 0
    pbar = tqdm.tqdm(total=len(prompt), disable=not verbose, desc="Batch Infer")
    outputs = []
    for p in prompt:
        pbar.set_postfix({"tokens_per_sec": f"{tokens_per_sec:.2f}"})
        start_time = time.time()
        result = infer(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            system_prompt=system_prompt,
            kv_cache=kv_cache,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            **kwargs,
        )
        end_time = time.time()
        kv_cache = result.pop("kv_cache")
        tokens_per_sec = result["output_len"] / (end_time - start_time)
        outputs.append(result)
        pbar.update(1)
    pbar.close()
    return outputs
