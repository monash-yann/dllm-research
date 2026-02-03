import time
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple
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
        self.outputs_all = []
        self.confidences_all = []
        self.transfer_idxs_all = []
        self.hidden_states_all = []
        self.record = {}

    def on_generate_start(self, prompt_len, **kwargs):
        self.prompt_len = prompt_len

    def on_step_end(self, 
                    x0:Tensor=None, 
                    confidences:Tensor=None, 
                    transfer_index:Tensor=None, 
                    hidden_states:Tuple[Tensor, ...]=None, 
                    **kwargs
                    ):
        if x0 is not None:
            self.outputs_all.append(x0[0].detach().cpu().numpy())
        if confidences is not None:
            self.confidences_all.append(confidences[0].detach().cpu().to(torch.float32).numpy())
        if transfer_index is not None:
            self.transfer_idxs_all.append(transfer_index[0].detach().cpu().numpy())
        if hidden_states is not None:
            np_h = np.array([h[0].detach().cpu().to(torch.float32).numpy() for h in hidden_states])  # (n_layers, seq_len, hidden_size)
            # print(f"Hidden states shape {np_h.shape}.")
            self.hidden_states_all.append(np_h)

    def on_generate_end(self, **kwargs):
        # 全部转为numpy数组返回
        self.record = {
            "prompt_len": self.prompt_len,
            "outputs_all": np.array(self.outputs_all),
            "confidences_all": np.array(self.confidences_all),
            "transfer_idxs_all": np.array(self.transfer_idxs_all),
            "hidden_states_all": np.array(self.hidden_states_all)
        }
