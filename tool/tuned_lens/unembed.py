import copy
import torch
import torch.nn as nn

from typing import cast
from transformers.modeling_utils import PreTrainedModel

def get_final_norm(model: PreTrainedModel) -> nn.Module:
    """
    获取模型的 Final Layer Norm 层。
    兼容本地加载和 trust_remote_code=True 动态加载的模型。
    """
    # 获取类名字符串，避免 import 依赖和类型不匹配问题
    model_type = type(model).__name__

    # 1. 检查 LLaDA 系列 (包括 LLaDAModel, LLaDAModelLM 等)
    if 'LLaDAModel' in model_type:
        # LLaDA 的标准结构: model.model.transformer.ln_f
        if hasattr(model, "model") and hasattr(model.model, "transformer") and hasattr(model.model.transformer, "ln_f"):
            return cast(nn.Module, model.model.transformer.ln_f)
    
    # 2. 检查 Dream 系列
    elif 'DreamModel' in model_type:
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
            
    # 3. 兜底策略：如果类名不匹配，尝试通过属性结构直接探测
    # 这是最稳健的方法，只要结构对就能跑
    if hasattr(model, "model"):
        # LLaDA style
        if hasattr(model.model, "transformer") and hasattr(model.model.transformer, "ln_f"):
            return cast(nn.Module, model.model.transformer.ln_f)
        # Dream/Standard style
        if hasattr(model.model, "norm"):
            return model.model.norm

    # 如果都找不到，抛出详细错误
    raise ValueError(f"Unsupported model type: {type(model)}. \n"
                     f"Could not locate final normalization layer (e.g., .model.transformer.ln_f).")


class Unembed(torch.nn.Module):
    """Module that maps transformer hidden states to logits (and vice versa)."""

    final_norm: nn.Module
    unembedding: torch.nn.Linear

    def __init__(self, model: PreTrainedModel):
        """Initialize unembed.

        Args:
            model: A HuggingFace model from which to extract the unembedding matrix.
        """
        super().__init__()
        
        # 使用增强版的 get_final_norm
        final_norm = get_final_norm(model)
        
        # 获取输出层 (LM Head)
        if not hasattr(model, "get_output_embeddings"):
             raise ValueError(f"Model {type(model)} does not support 'get_output_embeddings'.")
             
        unembedding_matrix = model.get_output_embeddings()
        
        if not isinstance(unembedding_matrix, torch.nn.Linear):
             raise ValueError(f"Expected linear output embeddings, got {type(unembedding_matrix)}")

        # 深拷贝以冻结参数，防止训练时意外更新
        self.final_norm = copy.deepcopy(final_norm)
        self.unembedding = copy.deepcopy(unembedding_matrix)

        # In general we don't want to finetune the unembed operation.
        self.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Convert hidden states into logits."""
        return self.unembedding(self.final_norm(h))