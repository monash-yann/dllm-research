from typing import Literal

import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

# from visualizer import get_local

from dllm.utils import add_gumbel_noise, set_seed
from dllm.tactics import DAEDAL
from dllm.DLLM import DLLM, DLLMConfig, GenerationMetrics, GenerateOutput
from dllm.recorder.recorder import MetricRecorder, StateTraceRecorder
from dataclasses import dataclass


@dataclass
class BaselineConfig(DLLMConfig):
    remasking: Literal["random", "low_confidence"] = "low_confidence"
    decoding_method: Literal["topk", "factor", "fixed"] = "topk"
    k:int = -1
    factor:float = 1.0
    confidence_threshold:float = 0.9


class DLLMBaseline(DLLM):
    """
        DLLMBaseline
        especially focusing on 'low-confidence' self.remasking
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: DLLMConfig,
    ) -> None:
        super().__init__(model, tokenizer, config)

    @torch.no_grad()
    def generate(
            self,
            prompt,
            gen_length=256,
            max_steps=256,
            block_length=256,
            records=['metrics'],
    ) -> GenerateOutput:

        batch = prompt.shape[0]
        prompt_len = prompt.shape[1]
        assert batch == 1, "Consider only batch_size = 1."

        metric_recorder = MetricRecorder()
        state_trace_recorder = StateTraceRecorder()
        if 'metrics' in records:
            metric_recorder.on_generate_start(max_steps=max_steps)
        if 'state_trace' in records:
            state_trace_recorder.on_generate_start(prompt_len=prompt_len)

        # assert gen_length <= self.config.max_gen_length, f"gen_length must <= max_gen_length({self.max_gen_length})"
        # assert max_steps <= self.config.max_steps, f"max_steps must <= model_max_steps({self.model_max_steps})"
        #
        # # initalize positional weights
        # if self.positional_weights_type == 'absolute':
        #     self.absolute_positional_weights = self.precompute_absolute_positional_weights(
        #         max_steps=max_steps, gen_length=gen_length, device=self.model.device, dtype=torch.float32
        #     )
        # elif self.positional_weights_type == 'ratio':
        #     pass
        # elif self.positional_weights_type == 'static':
        #     self.static_positional_weights = self.precompute_static_positional_weights(
        #         gen_length=gen_length, device=self.model.device, dtype=torch.float32
        #     )
        # else:
        #     pass

        # dynamic length
        adjusted_gen_lengths = self.length_strategy(self.model, prompt, self.config, gen_length)  # (b,)
        # attention mask!!!


        print("ajusted_gen_lengths's device:", adjusted_gen_lengths.device)
        # 向上取整到 block_length 的整数倍
        adjusted_gen_length = adjusted_gen_lengths.max().item()
        n_blocks = (adjusted_gen_length + block_length - 1) // block_length
        gen_length = n_blocks * block_length
        adjusted_steps = adjusted_gen_length
        block_steps = (gen_length) // n_blocks
        # assert max_steps % n_blocks == 0
        # block_steps = max_steps // n_blocks

        x = torch.full(
            (batch, prompt_len + gen_length), self.config.eos_id, dtype=torch.long
        ).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        cols = torch.arange(x.shape[1]).unsqueeze(0).to(adjusted_gen_lengths.device)  # (1, max_adjusted_seq_len)
        print("cols's device:", cols.device)
        mask_idxs = (cols >= prompt_len) & (cols < (prompt_len + adjusted_gen_lengths.unsqueeze(1)))  # (b, max_adjuested_seq_len)
        x[mask_idxs] = self.config.mask_id
        # print(f"adjusted_gen_length: {adjusted_gen_length}, gen_length: {gen_length}, n_blocks: {n_blocks}, n_mask_id: {(x == self.config.mask_id).sum().item()}.")
        prompt_index = (x != self.config.mask_id)

        print(f"decoding method: {self.decoding_method}, k={self.k}, factor={self.factor}, confidence_threshold={self.confidence_threshold}.")
        for num_block in range(n_blocks):
            block_start = prompt_len + num_block * block_length
            block_end = prompt_len + (num_block + 1) * block_length
            for i in range(block_steps):
                mask_index = (x == self.config.mask_id)
                # print(f"n_mask_id = {mask_index.sum().item()} at block {num_block}, step {i}.")
                if self.cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
                else:
                    # result = self.model(x, output_attentions=True)
                    logits = self.model(x).logits
                    # attentions = result.attentions
                    # print(attentions.shape)

                if self.dllm_type == 'llada':
                    pass
                elif self.dllm_type == 'dream':
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                x0 = torch.argmax(add_gumbel_noise(logits, temperature=self.temperature), dim=-1)  # (b, l)

                # demask & remask
                if self.remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)  # b, l, d_m
                    grab_index = torch.unsqueeze(x0, -1)
                    # x0_p (b,l) 即为选择的token_id的置信度(softmax概率)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=grab_index), -1)  # b, l
                elif self.remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(self.remasking)

                # x0_p[:, prompt_len + (num_block + 1) * block_length:] = -np.inf #semi-ar
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                confidence[:, 0: block_start] = confidence[:, block_end:] = -np.inf
                # print(f"n_remain_mask in current block: {(x[:, block_start: block_end] == self.mask_id).sum().item()}.")
                # print(f"n_positive_confidence: {(confidence[:, block_start: block_end] > 0).sum().item()}.")

                # applying positional weights dd
                if self.positional_weights_type == 'absolute':
                    confidence[:, prompt_len:] = confidence[:, prompt_len:] * self.absolute_positional_weights[
                        num_block * block_steps + i]
                elif self.positional_weights_type == 'ratio':
                    unmasked_ratio = (x[:, prompt_len:] != self.mask_id).sum().item() / gen_length
                    dynamic_positional_weights = self.compute_dynamic_positional_weights(gen_length, unmasked_ratio,
                                                                                         device=x0.device)
                    confidence[:, prompt_len:] = confidence[:, prompt_len:] * dynamic_positional_weights
                elif self.positional_weights_type == 'static':
                    confidence[:, prompt_len:] = confidence[:, prompt_len:] * self.static_positional_weights
                else:
                    pass

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                if self.remasking == 'low_confidence':
                    if self.decoding_method == 'factor':
                        # 根据Fast-dLLM中的公式: (n + 1) * (1 - c_{n}) < f 来确定最大的可并行解码n
                        # 1. 对>min_threshold的位置按confidence排序; 3. 对这些满足条件的index形成transfer_inedx
                        for b in range(confidence.shape[0]):
                            conf_b = confidence[b].clone()
                            cand_mask = (conf_b > 0)  # (L,)
                            # 根据cand_confs排序cand_idxs
                            cand_idxs = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)  # (n,)
                            cand_confs = conf_b[cand_mask]  # (n,)
                            sorted_order = torch.argsort(cand_confs, descending=True)
                            cand_idxs = cand_idxs[sorted_order]
                            cand_confs = cand_confs[sorted_order]
                            # 2. 从cand_confs最低conf处开始挨个试验可行的n，直到满足条件;
                            for conf_idx, conf in reversed(list(enumerate(cand_confs.tolist()))):
                                para_feasible_n = int(self.factor / (1 - conf + 1e-6) - 1)
                                #  3. 若满足公式，则根据这些满足条件的index形成transfer_inedx
                                if para_feasible_n >= conf_idx + 1:
                                    transfer_index[b].scatter_(dim=1, index=cand_idxs[:conf_idx + 1].unsqueeze(0), value=True)
                                    break
                    elif self.decoding_method == 'topk':  # default topk
                        k = self.k if self.k != -1 else block_length // block_steps
                        # print(f"in block {num_block}, step {i}, k={k}.")
                        for b in range(batch):
                            n_effective = (confidence[b] > 0).sum().item()
                            _, select_index = torch.topk(confidence[b], k=min(k, n_effective))
                            transfer_index[b, select_index] = True
                    elif self.decoding_method == 'fixed':
                        transfer_index = confidence > self.confidence_threshold   # maximum setting by fast-dllm
                    else:
                        pass
                    # top-1兜底. 若当前b的transfer_index全False, 且还有mask位置未解码完, 则选取最高confidence的位置进行解码
                    for b in range(batch):
                        if not transfer_index[b].any() and (x[b, block_start: block_end] == self.config.mask_id).any():
                            _, select_index = torch.topk(confidence[b], k=1)
                            transfer_index[b, select_index] = True

                x[transfer_index] = x0[transfer_index]
                # print(f"step: {accumulated_steps}, block: {num_block}, i: {i}, n_transferred: {transfer_index.sum().item()}.")

                # update recorder
                if 'metrics' in records:
                    metric_recorder.on_step_end()
                if 'state_trace' in records:
                    state_trace_recorder.on_step_end(x0, confidence, transfer_index)

                if not (x[:, block_start: block_end] == self.mask_id).any():
                    print(f"block {num_block} is decoded over in block_step_i={i}.")
                    break

        # compute recorder
        if 'metrics' in records:
            metric_recorder.on_generate_end(gen_length=adjusted_gen_length, max_steps=adjusted_steps)
        if 'state_trace' in records:
            state_trace_recorder.on_generate_end()

        return GenerateOutput(
            out=x,
            state_trace=state_trace_recorder.record,
            metrics=metric_recorder.record,
        )


def main():
    set_seed(1234)
    device = 'cuda:3'

    # 4-shot prompt
    # few_shot_filename = "../prompts/gsm8k_shot.txt"
    # prompts = []
    # with open(few_shot_filename, "r", encoding="utf-8") as f:
    #     for line in f:
    #         # python会把.txt中的字符当作原始字符串，此处转为普通字符串
    #         corrected_line = line.replace('\\n', '\n')
    #         prompts.append(corrected_line)
    # prompts = [codecs.decode(line, 'unicode_escape') for line in lines]

    # gsm8k prompt
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    prompts = gsm8k_dataset['test']['question'][1:2]

    # use llada
    model_path = "/home/xiangzhong_ayl/dllm/models/LLaDA-8B-Instruct"
    # model_path = "/homebck/home/xiangzhong_guest/dllm/models/LLADA-8B-Instruct"
    token_info = {
        'mask_id': 126336,
        'bos_id': 126080,
        'pad_id': 126081,
        'eos_id': 126081,
        'eot_id': 126348
    }

    # use dream
    # model_path = "../models/Dream-7B-Instruct"
    # token_info = {
    #     'mask_id': 151666,
    #     'bos_id': 151665,
    #     'pad_id': 151643,
    #     'eos_id': 151643,
    #     'eot_id': 151643
    # }

    config = BaselineConfig(
        cfg_scale=0.0,
        temperature=0.0,
        positional_weights_type='none',
        max_weight=1.0,
        initial_min_weight=0.05,
        remasking="low_confidence",
        decoding_method="topk",
        factor=1,
        k=1,
        confidence_threshold=0.9,
        **token_info
    )

    max_gen_steps = 64
    block_length = 64
    sampler = DLLMBaseline.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )
    sampler.set_length_strategy(DAEDAL())   # dynamic length

    tokenizer = sampler.tokenizer

    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)

        m = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        OUT = sampler.generate(prompt=input_ids, gen_length=max_gen_steps, max_steps=max_gen_steps, block_length=block_length, records=['metrics', 'state_trace'])
        out = OUT.out
        ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Prompt_{i}'s answer: {ans}\n")
        print(f"Generation Metrics: {OUT.metrics}\n")


if __name__ == '__main__':
    main()

