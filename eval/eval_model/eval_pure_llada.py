'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import json
import os
import sys
from dataclasses import fields, asdict
from typing import List

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from sampler.BaseSampler import GenerationMetrics, GenerateOutput
from sampler.PureLLaDASampler import PureLLaDASampler, PureLLaDASamplerConfig
from eval.eval_model.eval_base import set_seed, BaseEvalHarness


@register_model("eval_sampler")
class LLaDAEvalHarness(BaseEvalHarness):
    def __init__(
        self,
        model_path='./model_cache',
        batch_size=1,
        mc_num=128,
        steps=256,
        gen_length=256,
        block_length=256,
        device="cuda",
        **kwargs,
    ):

        sampler_config_fields = {f.name for f in fields(PureLLaDASamplerConfig)}
        sampler_kwargs = {
            key: kwargs[key]
            for key in sampler_config_fields
            if key in kwargs
        }
        sampler_config = PureLLaDASamplerConfig(**sampler_kwargs)

        sampler = PureLLaDASampler.from_path(
            model_path,
            config=sampler_config,
            torch_dtype=torch.bfloat16
        )

        super().__init__(model_path, batch_size, mc_num, steps, gen_length, block_length, sampler=sampler, **kwargs)


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
