import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
import torch.distributed as dist

from dllm.DLLM import DLLMConfig


def DAEDAL(eos_confidence_threshold=0.5, expansion_factor=8, eos_check_tokens=32):
    """
        eos_confidence_threshold: threshold to trigger expansion
        expansion_factor: expansion stride
        eos_check_tokens: window_size
    """
    print(f"[DAEDAL] initialized with: {locals()}.")

    # stage1 of DAEDAL: dynamic length
    @torch.no_grad()
    def daedal(model:PreTrainedModel, prompts:Tensor, config:DLLMConfig, initial_gen_length:int=64):
        """
            闭包函数.
            return: Tensor of shape (batch_size,) indicating the determined generation lengths for each sequence in the batch.
        """
        # the prompt here is already batch tokenized
        batch_size = prompts.shape[0]
        prompt_length = prompts.shape[1]
        device = prompts.device
        max_gen_length = config.max_gen_length
        is_main_process = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

        assert config.eos_id is not None
        if is_main_process:
            print(f"[DAEDAL] predicting lengths for batch_size={batch_size}...")
        gen_lengths = torch.full((batch_size,), initial_gen_length, dtype=torch.long, device=device)  # (b,)
        x = torch.full(
            (batch_size, prompt_length + initial_gen_length), config.mask_id, dtype=torch.long, device=device,
        )
        x[:, :prompt_length] = prompts.clone()
        while True:
            total_lengths = prompt_length + gen_lengths  # (b,)
            max_len_pre = x.shape[1]  # prev total_length
            arange_tensor_pre = torch.arange(max_len_pre, device=device).expand(batch_size, -1)  # (b, max_len_pre), act as index (0,1,...,max_len_pre-1)
            attention_mask_pre = (arange_tensor_pre < total_lengths.unsqueeze(1)).long()
            logits_pre = model(x, attention_mask=attention_mask_pre).logits
            batch_eos_confidences = _calculate_eos_confidence(logits_pre, total_lengths, prompt_length, eos_check_tokens, config.eos_id)  # (b,)
            del logits_pre
            sequences_to_expand = (batch_eos_confidences < eos_confidence_threshold) & (gen_lengths < max_gen_length)  # (b,)
            if not sequences_to_expand.any():
                if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                    print(
                        f"All sequences' EOS confidence reach the threshold {eos_confidence_threshold} or max length.")
                break
            # if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            #     print(
            #         f"Some sequences' EOS confidence ({[round(c.item(), 4) for c in batch_eos_confidences]}) < {eos_confidence_threshold}. Expand initial length.")
            new_gen_lengths = gen_lengths.clone()
            # enlarge
            new_gen_lengths[sequences_to_expand] = torch.clamp(gen_lengths[sequences_to_expand] + expansion_factor, max=max_gen_length)
            if new_gen_lengths.max() <= gen_lengths.max():
                if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                    print(
                        f"WARNING: Cannot expand initial length further (already at max length: {max_gen_length}).")
                break
            max_new_total_len = prompt_length + new_gen_lengths.max()
            new_x_tensor = torch.full((batch_size, max_new_total_len), config.eos_id, dtype=torch.long, device=device)
            for i in range(batch_size):
                original_total_len = prompt_length + gen_lengths[i].item()
                new_x_tensor[i, :original_total_len] = x[i, :original_total_len]
                if sequences_to_expand[i]:
                    new_total_len_i = prompt_length + new_gen_lengths[i].item()
                    new_x_tensor[i, original_total_len: new_total_len_i] = config.mask_id
            x = new_x_tensor
            gen_lengths = new_gen_lengths
        adjusted_gen_lengths = gen_lengths + int(eos_check_tokens / 2)
        adjusted_gen_lengths = torch.clamp(adjusted_gen_lengths, max=config.max_gen_length)
        print(f"[DAEDAL] determines lengths {adjusted_gen_lengths.tolist()}...")
        return adjusted_gen_lengths

    return daedal


def _calculate_eos_confidence(logits:Tensor, total_lengths:list, prompt_length:int, eos_check_tokens:int, eos_token_id:int):
    if eos_token_id is None:
        return torch.zeros(logits.shape[0], device=logits.device)
    confidences = F.softmax(logits, dim=-1)
    predicted_tokens = torch.argmax(logits, dim=-1)
    batch_eos_confidences = []
    for i in range(logits.shape[0]):
        eos_confs_for_avg = []
        start_scan_pos = total_lengths[i].item() - 1
        end_scan_pos = prompt_length - 1
        for pos in range(start_scan_pos, end_scan_pos, -1):
            if len(eos_confs_for_avg) >= eos_check_tokens:
                break
            if predicted_tokens[i, pos] == eos_token_id:
                eos_confs_for_avg.append(confidences[i, pos, eos_token_id].item())
        avg_conf = sum(eos_confs_for_avg) / eos_check_tokens
        batch_eos_confidences.append(avg_conf)
    return torch.tensor(batch_eos_confidences, device=logits.device)  # (b,)




