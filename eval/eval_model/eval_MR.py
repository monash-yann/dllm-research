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
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from sampler.MRSampler import MRSampler, MRSamplerConfig
from eval.eval_model.eval_base import set_seed, BaseEvalHarness


@register_model("eval_sampler")
class MRSamplerEvalHarness(BaseEvalHarness):
    def __init__(
            self,
            model_path: str = './model_cache',
            device="cuda",
            batch_size=1,
            mc_num=128,
            steps=256,
            gen_length=256,
            **kwargs,
    ):

        sampler_config_fields = {f.name for f in fields(MRSamplerConfig)}
        sampler_kwargs = {
            key: kwargs[key]
            for key in sampler_config_fields
            if key in kwargs
        }
        sampler_config = MRSamplerConfig(**sampler_kwargs)

        sampler = MRSampler.from_path(
            model_path,
            config=sampler_config
        )

        super().__init__(model_path, batch_size, mc_num, steps, gen_length, sampler, **kwargs)


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
