import codecs
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
class PureDLLMSamplerConfig(SamplerConfig):
    remasking: Literal["random", "low_confidence"] = "low_confidence"
    decoding_method: Literal["topk", "factor", "fixed"] = "topk"
    k:int = -1
    factor:float = 1.0
    confidence_threshold:float = 0.9


class PureDLLMSampler(BaseSampler):
    """
        PureDLLMSampler
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
            block_length=256,
            enable_metrics=True,
    ) -> GenerateOutput:

        assert gen_length <= self.model_max_genlength, f"gen_length must <= model_max_genlength({self.model_max_genlength})"
        assert max_steps <= self.model_max_steps, f"max_steps must <= model_max_steps({self.model_max_steps})"

        # initalize positional weights
        if self.positional_weights_type == 'absolute':
            self.absolute_positional_weights = self.precompute_absolute_positional_weights(
                max_steps=max_steps, gen_length=gen_length, device=self.model.device, dtype=torch.float32
            )
        elif self.positional_weights_type == 'ratio':
            pass
        elif self.positional_weights_type == 'static':
            self.static_positional_weights = self.precompute_static_positional_weights(
                gen_length=gen_length, device=self.model.device, dtype=torch.float32
            )
        else:
            pass

        # 主循环 (探索与加速)
        outputs = []
        confidences = []
        transfer_idxs = []
        phase_states = []  # [{'phase':'exploration/acceleration/mopup', 'range': (start, end)}]
        history_intervals_all = []  # [{'inceptive_step': 0, 'history_intervals': [[(start, end), ...], [(start, end), ...], ...]}]
        accumulated_steps = 0
        prompt_len = prompt.shape[1]

        start_time = time.perf_counter()

        x = torch.full((1, prompt_len + gen_length), self.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt_len] = prompt.clone()
        prompt_index = (x != self.mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert max_steps % num_blocks == 0
        block_steps = max_steps // num_blocks

        print(f"decoding method: {self.decoding_method}, k={self.k}, factor={self.factor}, confidence_threshold={self.confidence_threshold}.")
        for num_block in range(num_blocks):
                    # block_mask_index = (x[:, prompt_len + num_block * block_length: prompt_len + (
                    #     num_block + 1) * block_length:] == self.mask_id)
                    # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)  # 得到要demask的token数

            for i in range(block_steps):
                mask_index = (x == self.mask_id)
                if self.cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits
                if self.dllm_type == 'llada':
                    pass
                elif self.dllm_type == 'dream':
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)

                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                accumulated_steps += 1

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

                x0_p[:, prompt_len + (num_block + 1) * block_length:] = -np.inf #semi-ar

                x0 = torch.where(mask_index, x0, x)

                confidence = torch.where(mask_index, x0_p, -np.inf)

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
                                    transfer_index.scatter_(dim=1, index=cand_idxs[:conf_idx + 1].unsqueeze(0), value=True)
                                    break
                    elif self.decoding_method == 'topk':  # default topk
                        if self.k:
                            k = self.k
                        else:
                            assert block_length % block_steps == 0
                            k = block_length // block_steps
                        # print(f"in block {num_block}, step {i}, k={k}.")
                        for b in range(confidence.shape[0]):
                            n_effective = (confidence > 0).sum().item()
                            # print(f"=================n_effective: {n_effective}., k: {k}, selected_k:{min(k, n_effective)}=================")
                            _, select_index = torch.topk(confidence[b], k=min(k, n_effective))
                            transfer_index[b, select_index] = True
                            # print(f"select_index: {select_index.cpu().numpy()}.")
                    elif self.decoding_method == 'fixed':
                        transfer_index = confidence > self.confidence_threshold   # maximum setting by fast-dllm
                    else:
                        pass
                    # top-1兜底
                    if transfer_index.sum().item() == 0:
                        for b in range(confidence.shape[0]):
                            _, select_index = torch.topk(confidence[b], k=1)
                            transfer_index[b, select_index] = True

                x[transfer_index] = x0[transfer_index]
                # print(f"step: {accumulated_steps}, block: {num_block}, i: {i}, n_transferred: {transfer_index.sum().item()}.")

                # collecting states
                outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
                confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
                transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])

                if (x[:, prompt_len + num_block * block_length: prompt_len + (num_block+1) * block_length] == self.mask_id).sum().item() == 0:
                    print(f"block {num_block} is decoded over in block_step_i={i}.")
                    break

        # compute metrics
        total_steps = accumulated_steps

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"total steps: {total_steps}.")
        metrics = GenerationMetrics(
            use_seconds=duration,
            use_steps=total_steps,
            n_gen_tokens=gen_length,
            tokens_per_second=(gen_length / duration) if duration > 0 else 0,
            step_reduction_ratio=max_steps / accumulated_steps
        )
        print(metrics)

        return GenerateOutput(
            out=x,
            outputs=outputs,
            confidences=confidences,
            transfer_idxs=transfer_idxs,
            phase_states=phase_states,
            history_intervals_all=history_intervals_all,
            metrics=metrics,
        )


def main():
    set_seed(1234)
    device = 'cuda:1'
    model_path = "../models/LLaDA-8B-Instruct"

    # 4-shot prompt
    # few_shot_filename = "../prompts/gsm8k_shot.txt"
    # prompts = []
    # with open(few_shot_filename, "r", encoding="utf-8") as f:
    #     for line in f:
    #         # python会把.txt中的字符当作原始字符串，此处转为普通字符串
    #         corrected_line = line.replace('\\n', '\n')
    #         prompts.append(corrected_line)
    # prompts = [codecs.decode(line, 'unicode_escape') for line in lines]


    # base prompt
    # gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    # prompts = gsm8k_dataset['test']['question'][0:3]

    # base humaneval prompt
    humaneval_dataset = load_dataset('openai/openai_humaneval')
    prompts = humaneval_dataset['test']['prompt'][0:3]

    # use llada
    # model_path = "../models/LLaDA-8B-Instruct"
    # token_info = {
    #     'mask_id': 126336,
    #     'bos_id': 126080,
    #     'pad_id': 126081,
    #     'eos_id': 126081,
    #     'eot_id': 126348
    # }

    # use dream
    model_path = "../models/Dream-7B-Instruct"
    token_info = {
        'mask_id': 151666,
        'bos_id': 151665,
        'pad_id': 151643,
        'eos_id': 151643,
        'eot_id': 151643
    }

    config = PureDLLMSamplerConfig(
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

    max_gen_steps = 256
    block_length = 256
    sampler = PureDLLMSampler.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )

    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)
        tokenizer = sampler.tokenizer

        print(prompt_text)
        # m = [{"role": "user", "content": prompt_text}]
        # prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        # input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # print(prompt_text)

        OUT = sampler.generate(input_ids, gen_length=max_gen_steps, max_steps=max_gen_steps, block_length=block_length)
        out = OUT.out
        ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Prompt_{i}'s answer: {ans}\n")


if __name__ == '__main__':
    main()