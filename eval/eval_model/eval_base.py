'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import json
import os
import sys
from dataclasses import fields, asdict
from typing import List

import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from sampler.BaseSampler import BaseSampler, GenerateOutput, GenerationMetrics




def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseEvalHarness(LM):

    def __init__(
        self,
        model_path: str = './model_cache',
        batch_size=1,
        mc_num=128,
        steps=256,
        gen_length=256,
        sampler: BaseSampler = None,
        device="cuda",
        **kwargs,
    ):
        super().__init__()

        # 设置多gpu框架
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        if self.accelerator is not None:
            print(f"device_map: {self.accelerator.device}")
            # 将不在主线程的标准输出和标准错误重定向到空设备
            if not self.accelerator.is_main_process:
                f = open(os.devnull, 'w')
                sys.stdout = f
                sys.stderr = f

        # assign model to devices by accelerator
        sampler.model.eval()
        if self.accelerator is not None:
            self.sampler = self.accelerator.prepare(sampler)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        self.sampler.model = self.sampler.model.to(device)
        self.sampler.model.eval()
        self.device = self.sampler.model.device  # 从最终的模型获取最准确的设备信息

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.gen_length = gen_length
        self.steps = steps

        self.overall_metrics: List[GenerationMetrics] = []
        self.output_dir = kwargs['output_dir']

        self.is_instruct = True if 'instruct' in model_path.lower() else False

        print(f"!!! [DEBUG]: Running evaluation with args: \n{kwargs}")
        print(f"mc_num: {mc_num}\n"
              f"batch_size: {batch_size}\n"
              f"steps: {steps}\n"
              f"gen_length: {gen_length}\n"
              f"is_instruct: {self.is_instruct}\n")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.sampler.tokenizer(context + continuation)["input_ids"]
        context_enc = self.sampler.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    # PPL
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    # multiple choices
    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        on_main_process = self.accelerator is None or self.accelerator.is_main_process
        out = []
        with torch.no_grad():
            for i, elem in enumerate(tqdm(ds, desc="Computing likelihood..."), disable=on_main_process):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    # fixed answer
    def generate_until(self, requests: list[Instance]):

        on_main_process = self.accelerator is None or self.accelerator.is_main_process
        tokenizer = self.sampler.tokenizer
        out = []
        for req in tqdm(requests, desc="Generating...", disable=not on_main_process):
            question = req.args[0]
            # treat as base_model on in humaneval dataset
            if (not self.is_instruct) or ('task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval')):
            # if (not self.is_instruct):
                prompt_str = question
            else:
                m = [{"role": "user", "content": question}]
                prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            prompt = tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.device)
            # print(f"\n{'=' * 20} prompt_str: \n{prompt_str} {'=' * 20}")

            stop_tokens = req.args[1]['until']
            stop_tokens.append(tokenizer.eos_token)

            # print('#' * 20 + f"the prompt is: {elem['question_text']}" + '#' * 20)
            OUT: GenerateOutput = self.sampler.generate(prompt, gen_length=self.gen_length, max_steps=self.steps)
            generated_answer = OUT.out
            generated_answer = tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            # print('#' * 20 + f"generated_answer: {generated_answer}" + '#' * 20)
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]
            if on_main_process:
                print('#' * 20 + f"generated_answer after spliting: {generated_answer}" + '#' * 20)

            # remove special tokens
            generated_answer_ids = tokenizer(generated_answer)["input_ids"]
            generated_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            # accumulate metrics
            metrics = OUT.metrics
            self.overall_metrics.append(metrics)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def __del__(self):
        """
        析构函数，在评估任务结束、对象被销毁时自动调用。
        用于聚合所有自定义指标并写入JSON文件。
        """
        # 仅在主进程上执行聚合和写入操作，避免多GPU时重复写入
        overall_metrics: List[GenerationMetrics] = []
        if self.accelerator is not None:
            print(f"[Info] Collecting metrics......")
            overall_metrics = self.accelerator.gather_for_metrics(self.overall_metrics)

            if self._rank != 0:
                return
            print(f"[Info] Gathered {len(overall_metrics)} metrics in total")

        if not overall_metrics:
            print("No overall metrics were collected. Skipping report generation.")
            return

        print(f"[Info] Computing metrics......")
        total_use_seconds = 0
        total_use_steps = 0
        total_n_gen_tokens = 0
        for metric in overall_metrics:
            total_use_seconds += metric.use_seconds
            total_use_steps += metric.use_steps
            total_n_gen_tokens += metric.n_gen_tokens

        summary_metrics = GenerationMetrics(
            use_seconds=total_use_seconds,
            use_steps=total_use_steps,
            n_gen_tokens=total_n_gen_tokens,
            tokens_per_second=(total_n_gen_tokens / total_use_seconds) if total_use_seconds > 0 else 0,
            step_reduction_ratio=len(overall_metrics) * self.steps / total_use_steps if total_use_steps > 0 else 0
        )
        metrics_report = {
            "summary": asdict(summary_metrics),
            "per_sample": [asdict(metric) for metric in overall_metrics]
        }

        metrics_fpath = os.path.join(self.output_dir, "overall_metrics.json")
        with open(metrics_fpath, 'w', encoding='utf-8') as f:
            json.dump(metrics_report, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
