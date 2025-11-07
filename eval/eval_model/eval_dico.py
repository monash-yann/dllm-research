'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
from dataclasses import fields

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from sampler.DiCoSampler import DiCoSampler, DiCoSamplerConfig
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
            block_length=256,
            **kwargs,
    ):

        sampler_config_fields = {f.name for f in fields(DiCoSamplerConfig)}
        sampler_kwargs = {
            key: kwargs[key]
            for key in sampler_config_fields
            if key in kwargs
        }
        sampler_config = DiCoSamplerConfig(**sampler_kwargs)

        sampler = DiCoSampler.from_path(
            model_path,
            config=sampler_config,
            torch_dtype=torch.bfloat16
        )

        super().__init__(model_path, batch_size, mc_num, steps, gen_length, block_length, sampler=sampler, **kwargs)


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
