from typing import List, Tuple, Dict, Any, Literal
from torch import Tensor

import torch
import time
import math
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from visualizer import get_local

from sampler.utils import add_gumbel_noise, get_num_transfer_tokens, set_seed
from sampler.BaseSampler import BaseSampler, SamplerConfig, GenerationMetrics, GenerateOutput
from dataclasses import dataclass, fields, asdict, field

@dataclass
class LLaDASamplerConfig(SamplerConfig):
    block_length: int = 256
    remasking: Literal["random", "low_confidence"] = "low_confidence"

@dataclass
class GenerationMetrics:
    """用于存储单次生成过程的性能指标"""
    use_seconds: float
    use_steps: int
    n_gen_tokens: int
    tokens_per_second: float
    step_reduction_ratio: float

@dataclass
class GenerateOutput:
    out: torch.Tensor
    metrics: GenerationMetrics
    # 以下为用于调试和分析的详细过程数据，默认为空列表，以防不需要时占用内存
    outputs: List[np.ndarray] = field(default_factory=list)
    confidences: List[np.ndarray] = field(default_factory=list)
    transfer_idxs: List[np.ndarray] = field(default_factory=list)
    phase_states: List= field(default_factory=list)
    exploration_intervals: List = field(default_factory=list)


class PureLLaDASampler(BaseSampler):
    """
        PureLLaDASampler
        especially focusing on 'low-confidence' self.remasking
    """
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: SamplerConfig,
    ) -> None:
        super().__init__(model, tokenizer, config)


    @torch.no_grad()
    def generate(
        self,
        prompt,
        gen_length=256,
        max_steps=256,
        enable_metrics=True,
    ) -> GenerateOutput:
        """
        实现“多区域并行置信度驱动解码”思路的主函数。
        """
        # 初始化

        assert gen_length <= self.model_max_genlength, f"gen_length must <= model_max_genlength({self.model_max_genlength})"
        assert max_steps <= self.model_max_steps, f"max_steps must <= model_max_steps({self.model_max_steps})"

        # 主循环 (探索与加速)
        outputs = []
        confidences = []
        transfer_idxs = []
        phase_states = []  # [{'phase':'exploration/acceleration/mopup', 'range': (start, end)}]
        exploration_intervals = []  # [{'inceptive_step': 0, 'history_intervals': [[(start, end), ...], [(start, end), ...], ...]}]
        accumulated_steps = 0

        start_time = time.perf_counter()

        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = (x != self.mask_id)

        assert gen_length % self.block_length == 0
        num_blocks = gen_length // self.block_length

        assert max_steps % num_blocks == 0
        max_steps = max_steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * self.block_length: prompt.shape[1] + (
                        num_block + 1) * self.block_length:] == self.mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, max_steps)  # 得到要demask的token数

            for i in range(max_steps):
                mask_index = (x == self.mask_id)
                if self.cfg_scale > 0.:
                    # un_x: 遮盖住prompt的无条件预测
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    # 预测结果 = 无条件预测结果 + 影响权重 * 条件引导方向
                    logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                # gumbel_noise+argmax: 一种采样技术。经证明gumbel_noise会影响argmax的操作，使对原softmax-logits变为以概率p取置信度索引而不是100%取最大置信度的索引。
                #   经gumbel_noise后的logits会受到扰乱，因此还需保留原本的logits以做后续的softmax置信度分析
                logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                # demask & remask
                if self.remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)  # b, l, d_m
                    grab_index = torch.unsqueeze(x0, -1)
                    # x0_p (b,l) 即为选择的token_id的置信度(softmax概率)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=grab_index), -1)  # b, l
                    # 此处输出回合全部softmax分数(包括non-mask部分)

                elif self.remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(self.remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * self.block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)  # demask

                confidence = torch.where(mask_index, x0_p, -np.inf)
                # 此处输出当前step中mask处的预测置信度

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        # compute metrics
        end_time = time.perf_counter()
        duration = end_time - start_time

        metrics = GenerationMetrics(
            use_seconds=duration,
            use_steps=max_steps,
            n_gen_tokens=gen_length,
            tokens_per_second=(gen_length / duration) if duration > 0 else 0,
            step_reduction_ratio=0
        )
        print(metrics)

        return GenerateOutput(
            out=x,
            outputs=outputs,
            confidences=confidences,
            transfer_idxs=transfer_idxs,
            phase_states=phase_states,
            exploration_intervals=exploration_intervals,
            metrics=metrics,
        )


def main():
    set_seed(1234)
    device = 'cuda:1'
    model_path = "../models/LLaDA-8B-Instruct"

    # 4-shot prompt
    few_shot_filename = "../prompts/gsm8k_shot.txt"
    with open(few_shot_filename, "r", encoding="utf-8") as f:
        prompts= f.readlines()[0:3]

    # base prompt
    # gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    # prompts = gsm8k_dataset['test']['question'][2:3]

    # --- 使用类进行生成 ---
    config = LLaDASamplerConfig(
        block_length=256,
        remasking="low_confidence"
    )

    max_gen_steps = 256
    sampler = PureLLaDASampler.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )

    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)
        tokenizer = sampler.tokenizer

        m = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

        OUT = sampler.generate(input_ids, gen_length=max_gen_steps, max_steps=max_gen_steps)
        out = OUT.out
        ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Prompt_{i}'s answer: {ans}\n")


if __name__ == '__main__':
    main()