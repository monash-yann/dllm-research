import time
from dataclasses import dataclass, field
from typing import List, Any, Dict
import numpy as np
import torch
from torch import Tensor
from dllm.DLLM import GenerationMetrics


class CallbackTemplate:
    def on_generate_start(self, **kwargs): pass

    def on_step_end(self, **kwargs): pass

    def on_generate_end(self, **kwargs): pass


class MetricRecorder(CallbackTemplate):
    """专门负责计算 TPS, SRR, Time 等性能指标"""

    def __init__(self):
        self.start_time = 0
        self.record = None
        self.accumulated_steps = 0

    def on_generate_start(self, **kwargs):
        self.start_time = time.perf_counter()
        self.accumulated_steps = 0

    def on_step_end(self, **kwargs):
        self.accumulated_steps += 1

    def on_generate_end(self, gen_length, max_steps, **kwargs):
        end_time = time.perf_counter()
        duration = end_time - self.start_time

        # 这里的 GenerationMetrics 定义和你原代码一致
        self.record = GenerationMetrics(
            use_seconds=duration,
            use_steps=self.accumulated_steps,
            n_gen_tokens=gen_length,
            tokens_per_second=(gen_length / duration) if duration > 0 else 0,
            step_reduction_ratio=max_steps / self.accumulated_steps if self.accumulated_steps > 0 else 0
        )
        # print(f"[Callback] Metrics Computed: {self.metrics}")


class StateTraceRecorder(CallbackTemplate):
    """专门负责记录中间状态，用于可视化 (原代码中 append 到 outputs/confidences 的逻辑)"""

    def __init__(self):
        self.prompt_len = 0
        self.outputs = []
        self.confidences = []
        self.transfer_idxs = []
        self.record = {}

    def on_generate_start(self, prompt_len, **kwargs):
        self.prompt_len = prompt_len

    def on_step_end(self, x: Tensor, confidence: Tensor, transfer_idx: Tensor, **kwargs):
        self.outputs.append(x.detach().cpu().numpy()[0][self.prompt_len:])
        self.confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][self.prompt_len:])
        self.transfer_idxs.append(transfer_idx.detach().cpu().numpy()[0][self.prompt_len:])

    def on_generate_end(self, **kwargs):
        self.record = {
            "outputs": self.outputs,
            "confidences": self.confidences,
            "transfer_idxs": self.transfer_idxs
        }
