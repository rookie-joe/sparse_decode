# -*- encoding: utf-8 -*-
import torch
import transformers
from transformers import AutoTokenizer
from infer_utils.infer import batch_infer
from infer_utils.modeling_qwen2 import Qwen2ForCausalLM


if __name__ == "__main__":
    transformers.set_seed(42)

    model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    llm = Qwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = """Please translate the following Chinese text to English. \n\nChinese Text:\n去年10月，希腊伯罗奔尼撒半岛发现了传说中的泰涅亚城遗址。 “泰涅亚城（Tenea）就埋在这底下，”她告诉我。那时的古城建筑高于海平面，有凉爽的微风，所以度夏的行宫很可能建在这里。她指了指一家坐落在独特的、几乎是方形山脚下的传统餐厅。“这个餐厅的位置当时就是个水磨，”她说。 科尔卡是希腊裔美国人，也是顶尖的考古学家。她最近公布了她40年职业生涯中最大的发现。泰涅亚这座失落之城，在许多希腊神话和历史文献中被提到，比如俄狄浦斯（Oedipus）的古老传说。传说中这位底比斯（Thebes）国王无意中杀死了他的父亲，娶了他的母亲。如今这座古城遗址在去年十月被科尔卡团队发现，就在我们开车经过的这片土地之下。  \n\nTranslation:\n"""

    prompts = [prompt] * 10

    cache_kwargs = dict(
        kv_cache=None,
        max_length=8192,
    )

    generate_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        top_k=0,
        top_p=0.85,
        temperature=0.2,
    )

    result = batch_infer(
        llm,
        tokenizer,
        prompts,
        **cache_kwargs,
        **generate_kwargs,
    )
