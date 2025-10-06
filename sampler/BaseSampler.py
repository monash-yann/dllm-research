from abc import abstractmethod
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
from dataclasses import dataclass, fields, asdict, field

@dataclass
class SamplerConfig:
    # Model forward config
    cfg_scale: float = 0.0
    temperature: float = 0.0
    # Special token ids
    mask_id: int = 126336
    endoftext_id: int = 126081
    eot_id: int = 126348
    # Generation general config
    model_max_genlength: int = 2048
    model_max_steps: int = 2048

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


class BaseSampler:
    """
        An Abstract Class for all Samplers
    """
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: SamplerConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device

        # binding configs to sampler
        for field in fields(config):
            print(f"{field.name}: {getattr(config, field.name)}")
            setattr(self, field.name, getattr(config, field.name))

    @classmethod
    def from_path(
            cls,
            model_path: str,
            config: SamplerConfig,
            device: str | None = None,
            torch_dtype: torch.dtype = torch.bfloat16,
    ):
        print(f"Loading model and tokenizer from path: {model_path}")

        # get_local.activate()  # 在引入模型之前，激活装饰器
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        if device is not None:
            model.to(device=device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        return cls(model=model, tokenizer=tokenizer, config=config)

    @torch.no_grad()
    def _model_forward(
            self,
            x: torch.Tensor,
            prompt_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = self.mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits_batch = self.model(x_).logits
            logits, un_logits = torch.chunk(logits_batch, 2, dim=0)
            logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
        else:
            logits = self.model(x).logits

        logits_with_noise = add_gumbel_noise(logits, self.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        mask_index = (x == self.mask_id)
        confidence = torch.where(mask_index, x0_p, -np.inf)
        return x0, confidence, x0_p

    @torch.no_grad()
    @abstractmethod
    def generate(
        self,
        prompt,
        gen_length=256,
        max_steps=256,
        enable_metrics=True,
    ) -> GenerateOutput:
        pass