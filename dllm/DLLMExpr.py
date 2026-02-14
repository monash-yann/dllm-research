import codecs
from typing import Literal

import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

# from visualizer import get_local

from dllm.DLLMBaseline import BaselineConfig
from dllm.utils import add_gumbel_noise, set_seed
from dllm.tactics import DAEDAL
from dllm.DLLM import DLLM, DLLMConfig, GenerationMetrics, GenerateOutput
from dllm.recorder.recorder import MetricRecorder, StateTraceRecorder
from dataclasses import dataclass


class DLLMExpr(DLLM):
    """
        DLLMExpr
        especially focusing on 'low-confidence' remasking
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
        config = self.config
        assert gen_length <= config.max_gen_length, f"gen_length must <= max_gen_length({config.max_gen_length})"
        assert max_steps <= config.max_steps, f"max_steps must <= max_steps({config.max_steps})"

        print(f"decoding method: {config.decoding_method}, k={config.k}, factor={config.factor}, confidence_threshold={config.confidence_threshold}.")

        batch = prompt.shape[0]
        prompt_len = prompt.shape[1]
        assert batch == 1, "Consider only batch_size = 1."

        metric_recorder = MetricRecorder()
        state_trace_recorder = StateTraceRecorder()
        if 'metrics' in records:
            metric_recorder.on_generate_start()
        if 'state_trace' in records:
            state_trace_recorder.on_generate_start(prompt_len=prompt_len)

        # dynamic length
        batch_gen_lengths = self.length_strategy(self.model, prompt, config, gen_length)  # (b,)
        gen_length = batch_gen_lengths.max().item()

        total_lengths = prompt_len + batch_gen_lengths  # (b,), 曾经踩坑忘记算prompt_len了        
        n_blocks = (gen_length + block_length - 1) // block_length
        print(f"adjusted gen_length: {gen_length}, n_blocks: {n_blocks}.")
        
        x = torch.full(
            (batch, prompt_len + gen_length), config.eos_id, dtype=torch.long
        ).to(self.model.device)
        x[:, :prompt_len] = prompt.clone()

        # 创造x、设置mask_id、生成attention_mask
        for b in range(batch):
            x[b, prompt_len: total_lengths[b]] = config.mask_id
        arange_tensor = torch.arange(x.shape[1], device=x.device).expand(batch, -1)
        attention_mask = (arange_tensor < total_lengths.unsqueeze(1)).long()    #只对有效长度内的token进行attention
        prompt_index = (x != config.mask_id)
        
        # curr_decoding_pos: 记录每个batch当前解码到的位置
        curr_decoding_pos = torch.full((batch,), prompt_len, dtype=torch.long, device=x.device) #(b,)
        mask_token_index = (x == config.mask_id)
        while mask_token_index.any():
            # 预处理，统计信息
            block_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for b in range(batch):
                block_mask[b, curr_decoding_pos[b]: min(curr_decoding_pos[b] + block_length, total_lengths[b].item())] = True
        
            # 模型传播
            if config.cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = config.mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = self.model(x_, attention_mask=torch.cat([attention_mask, attention_mask], dim=0)).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (config.cfg_scale + 1) * (logits - un_logits)
            else:
                # result = (x, output_attentions=True)
                output = self.model(x, attention_mask=attention_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = torch.stack(output.hidden_states)  # will be collected by recorder. n_layers * (b, seq_len, hidden_size)
            
            if config.dllm_type == 'llada':
                pass
            elif config.dllm_type == 'dream':
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            x0 = torch.argmax(add_gumbel_noise(logits, temperature=config.temperature), dim=-1)  # (b, l)
            p = F.softmax(logits, dim=-1)  # generated confidences: (b, seq_len, vocab_size)
            confidences = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1) #(b, seq_len)            

            # 解码策略
            effective_confidences = confidences.masked_fill(~(block_mask & mask_token_index), -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for b in range(batch):
                if curr_decoding_pos[b] >= total_lengths[b]: continue   # 当前seq已解完则等其它
                block_start, block_end = curr_decoding_pos[b], min(curr_decoding_pos[b] + block_length, total_lengths[b].item())
                if config.decoding_method == 'factor':
                    # 根据Fast-dLLM中的公式: (n + 1) * (1 - c_{n}) < f 来确定最大的可并行解码n
                    # 1. 对>min_threshold的位置按confidence排序; 3. 对这些满足条件的index形成transfer_inedx
                    conf_b = effective_confidences[b].clone()
                    cand_mask = (conf_b > 0)  # (L,)
                    # 根据cand_confs排序cand_idxs
                    cand_idxs = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)  # (n,)
                    cand_confs = conf_b[cand_mask]  # (n,)
                    sorted_order = torch.argsort(cand_confs, descending=True)
                    cand_idxs = cand_idxs[sorted_order]
                    cand_confs = cand_confs[sorted_order]
                    # 2. 从cand_confs最低conf处开始挨个试验可行的n，直到满足条件;
                    for conf_idx, conf in reversed(list(enumerate(cand_confs.tolist()))):
                        para_feasible_n = int(config.factor / (1 - conf + 1e-6) - 1)
                        #  3. 若满足公式，则根据这些满足条件的index形成transfer_inedx
                        if para_feasible_n >= conf_idx + 1:
                            transfer_index[b, cand_idxs[:conf_idx + 1]] = True
                            break
                elif config.decoding_method == 'fixed':
                    transfer_index[b] = effective_confidences[b] > config.confidence_threshold   # maximum setting by fast-dllm
                elif config.decoding_method == 'topk':  # default topk
                    if config.k <= 0:
                        raise ValueError("k must be a positive integer.")
                    n_effective = (effective_confidences[b] > 0).sum().item()
                    _, select_index = torch.topk(effective_confidences[b], k=min(config.k, n_effective))
                    transfer_index[b, select_index] = True
                else:
                    pass

                # top-1兜底.
                if not transfer_index[b].any() and (x[b, block_start: block_end] == config.mask_id).any():
                    _, select_index = torch.topk(effective_confidences[b], k=1)
                    transfer_index[b, select_index] = True

            # 更新信息
            x[transfer_index] = x0[transfer_index]
            mask_token_index = (x == config.mask_id)

            for b in range(batch):
                if curr_decoding_pos[b] >= total_lengths[b]: continue
                block_start, block_end = curr_decoding_pos[b], min(curr_decoding_pos[b] + block_length, total_lengths[b].item())
                if (x[b, block_start: block_end] == config.mask_id).any(): continue
                curr_decoding_pos[b] = block_end

            # update recorder
            if 'metrics' in records:
                # print(f"step {metric_recorder.accumulated_steps} over")
                metric_recorder.on_step_end()
            if 'state_trace' in records:
                state_trace_recorder.on_step_end(x0, effective_confidences, transfer_index, hidden_states)


        # compute recorder
        if 'metrics' in records:
            metric_recorder.on_generate_end(gen_length=gen_length, max_steps=gen_length)
        if 'state_trace' in records:
            state_trace_recorder.on_generate_end()


        # 把steps_hidden_states存到本地文件
        # np.save(f'visualization/output/hidden_states_baseline.npy', np.array(steps_hidden_states))

        return GenerateOutput(
            out=x,
            state_trace=state_trace_recorder.record,
            metrics=metric_recorder.record,
        )


def main():
    # set_seed(1234)
    device = 'cuda:1'

    # gsm8k prompt
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    prompts = gsm8k_dataset['test']['question'][0:1]

    # 4-shot prompt
    # few_shot_filename = "prompts/gsm8k_shot.txt"
    # prompts = []
    # with open(few_shot_filename, "r", encoding="utf-8") as f:
    #     for line in f:
    #         # python会把.txt中的字符当作原始字符串，此处转为普通字符串
    #         corrected_line = line.replace('\\n', '\n')
    #         prompts.append(corrected_line)
    # prompts = [codecs.decode(line, 'unicode_escape') for line in prompts]

    # humaneval_dataset = load_dataset("openai/openai_humaneval", split="test")
    # prompts = humaneval_dataset['prompt'][0:3]

    # use llada
    model_path = "/home/xiangzhong_ayl/dllm/models/LLaDA-8B-Instruct"
    # _path = "/homebck/home/xiangzhong_guest/dllm/s/LLADA-8B-Instruct"
    token_info = {
        'mask_id': 126336,
        'bos_id': 126080,
        'pad_id': 126081,
        'eos_id': 126081,
        'eot_id': 126348
    }

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

    gen_length = 32
    block_length = 32
    sampler = DLLMExpr.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )
    # sampler.set_length_strategy(DAEDAL())   # dynamic length

    tokenizer = sampler.tokenizer
    
    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)
        print(f"Prompt_{i}: {prompt_text}\n")

        m = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        OUT = sampler.generate(prompt=input_ids, gen_length=gen_length, max_steps=gen_length, block_length=block_length, records=['metrics', 'state_trace'])
        out = OUT.out
        ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        print(f"Prompt_{i}'s answer: {ans}\n")
        print(f"Generation Metrics: {OUT.metrics}\n")
        print(f"hidden_states shape: {OUT.state_trace['hidden_states_all'].shape}\n")

        # 将hidden states保存到本地文件
        # np.save(f'visualization/huashan2/rawdata/hidden_states_gsm8k_pmt{i}.npy', OUT.state_trace['hidden_states_all'])


if __name__ == '__main__':
    main()

