from collections import defaultdict
import torch
from torch import Tensor


def sink_metric_epsilon(W, epsilon=0.0): 
    """
    Paper's sink rule with epsilon sensitivity.

    W: (T, L, H, S, S) attention tensor
    epsilon: additive threshold
    return: (T, S) and (T, L, S)
    """

    W = torch.tensor(W)  # ensure it's a tensor
    # mean 和 sum 都是线性操作，顺序不影响结果
    T, L, H, S, _ = W.shape
    incoming = W.sum(dim=-2)    # (T, L, H, S) sum over queries. 
    S_eff = incoming.shape[-1]
    denom = max(S_eff - 1, 1)

    incoming_head_avg = incoming.mean(dim=-2)  # (T, L, S) 
    total_head_avg = incoming_head_avg.sum(dim=-1, keepdim=True)  # (T, L, 1)
    others_mean_head_avg = (total_head_avg - incoming_head_avg) / denom    # (T, L, S)
    delta_head_avg = incoming_head_avg - others_mean_head_avg  # (T, L, S)
    sink_mask_head_avg = delta_head_avg > epsilon       # (T, L, S) 

    incoming_layer_head_avg = incoming_head_avg.mean(dim=-2)  # (T, S)
    total_layer_head_avg = incoming_layer_head_avg.sum(dim=-1, keepdim=True)  # (T, 1)
    others_mean_layer_head_avg = (total_layer_head_avg - incoming_layer_head_avg) / denom    # (T, S)
    delta_layer_head_avg = incoming_layer_head_avg - others_mean_layer_head_avg  # (T, S)
    sink_mask_layer_head_avg = delta_layer_head_avg > epsilon       # (T, S)

    return {
        "sink_mask_head_avg": sink_mask_head_avg,
        "sink_mask_layer_head_avg": sink_mask_layer_head_avg,
    } 











# def sink_metric_epsilon_steps(W, epsilon=0.0, exclude_self=False):
#     """
#     Paper's sink rule with epsilon sensitivity.

#     W: (T, L, H, S, S) attention tensor
#     epsilon: additive threshold
#     exclude_self: if True, zero the diagonal and renormalize per row
#     return: (T, )
#     """
#     W = torch.tensor(W)  # ensure it's a tensor

#     T, L, H, S, _ = W.shape
#     A = W.mean(dim=(1, 2))  # (T, S, S) average over layers & heads

#     # mask self-attention and renormalize per row (per-step)
#     if exclude_self:
#         I = torch.eye(S, device=A.device, dtype=A.dtype).unsqueeze(0)  # (1, S, S)，单位矩阵
#         A = A * (1.0 - I)
#     # row-normalize (rows are queries; sum over columns -> 1) [这步在exclude_self=Flase时有些多余，可能是为了保持数值稳定性]
#     row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1e-12) # (1, S, 1) 
#     A = A / row_sums 

#     # Incoming attention per step & position, then average across steps
#     incoming = A.sum(dim=-2)        # (T, S) sum over queries. 

#     # Mean of "other tokens" for each j. [根据公式求sink位置]
#     total = incoming.sum(dim=-1, keepdim=True)  # (T, 1) sum over all tokens
#     S_eff = incoming.shape[-1]
#     denom = max(S_eff - 1, 1)
#     others_mean = (total - incoming) / denom    # (T, S) mean of other tokens

#     delta = incoming - others_mean  # (T, S) difference between incoming and others' mean
#     sink_mask = delta > epsilon       # (T, S) boolean mask where True indicates a sink position

#     return {
#         "incoming": incoming, "others_mean": others_mean, "delta": delta,
#         "sink_mask": sink_mask
#     }


# def sink_metric_epsilon_steps_layers(W, epsilon=0.0, exclude_self=False):
#     """
#     Paper's sink rule with epsilon sensitivity.

#     W: (T, L, H, S, S) attention tensor
#     epsilon: additive threshold
#     exclude_self: if True, zero the diagonal and renormalize per row
#     return: (T, L)
#     """
#     W = torch.tensor(W)  # ensure it's a tensor

#     T, L, H, S, _ = W.shape
#     A = W.mean(dim=2)  # (T, L, S, S) average over heads

#     # mask self-attention and renormalize per row (per-step)
#     if exclude_self:
#         I = torch.eye(S, device=A.device, dtype=A.dtype)  # (S, S)，单位矩阵
#         A = A * (1.0 - I)   # (T, L, S, S) zero out diagonal
#     # row-normalize (rows are queries; sum over columns -> 1) [这步在exclude_self=Flase时有些多余，可能是为了保持数值稳定性]
#     row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1e-12) # (1, S, 1) 
#     A = A / row_sums 

#     # Incoming attention per step & position, then average across steps
#     incoming = A.sum(dim=-2)        # (T, L, S) sum over queries. 

#     # Mean of "other tokens" for each j. [根据公式求sink位置]
#     total = incoming.sum(dim=-1, keepdim=True)  # (T, L, 1) sum over all tokens
#     S_eff = incoming.shape[-1]
#     denom = max(S_eff - 1, 1)
#     others_mean = (total - incoming) / denom    # (T, L, S) mean of other tokens

#     delta = incoming - others_mean  # (T, L, S) difference between incoming and others' mean
#     sink_mask = delta > epsilon       # (T, L, S) boolean mask where True indicates a sink position

#     return {
#         "incoming": incoming, "others_mean": others_mean, "delta": delta,
#         "sink_mask": sink_mask
#     }
