from typing import List, Literal
from torch import Tensor

import torch
import time
import math
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

from sampler.BaseSampler import BaseSampler, SamplerConfig, GenerationMetrics, GenerateOutput
from sampler.utils import add_gumbel_noise, get_num_transfer_tokens, set_seed
from dataclasses import dataclass, fields, asdict, field


@dataclass
class MRSamplerConfig(SamplerConfig):
    """
    用于存储 MRSampler 所有超参数的数据结构。
    """
    # Exploration phase config
    max_exploration_steps: int = 10
    exploration_N: int = 2
    exploration_M: int = 3
    exploration_threshold: float = 0.2
    # Acceleration phase config
    acceleration_parallel_method: Literal["fixed", "factor", "entropy", "margin"] = "fixed"
    acceleration_threshold: float = 0.8
    acceleration_low_threshold: float = 0.6
    acceleration_factor: float = 1
    # Mop-up phase config
    mopup_gate_ratio: float = 0.8
    mopup_margin_threshold: float = 5.0
    max_mopup_steps: int = 10
    mopup_speed: int = 2


class MRSampler(BaseSampler):
    """
        DiCo Sampler：
        1. Divide Phase: construct decoding zones with stable and moderate confidence, do "gentle" parallel decoding.
        2. Conquer Phase: do confidence-based parallel decoding on decoding zones.
        3. Finalize Phase: do margin-based parallel+top1 decoding on global context。
    """
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: MRSamplerConfig
    ) -> None:
        super().__init__(model, tokenizer, config)


    def _merge_intervals(self, intervals: List[tuple[int, int]]) -> List[tuple[int, int]]:
        """
            merge overlapping intervals
            eg: [(10, 20), (15, 25), (40, 50)] -> [(10, 25), (40, 50)]
        """
        if len(intervals) < 2:
            return intervals
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged_intervals = [sorted_intervals[0]]
        for current_start, current_end in sorted_intervals[1:]:
            mem_start, mem_end = merged_intervals[-1]
            if mem_end + 1 >= current_start:
                merged_intervals[-1] = (mem_start, max(mem_end, current_end))
            else:
                merged_intervals.append((current_start, current_end))
        return merged_intervals


    #  core methods  #
    def exploration_phase(
            self,
            x: Tensor,
            prompt_index: Tensor,
            exp_steps: int = 0,
            current_step: int = -1,
    ):
        """
        Divide Phase: construct decoding zones with stable and moderate confidence, do "gentle" parallel decoding.
            exp_steps: maximum exploratory iterations
        """
        prompt_len = prompt_index[0].sum().item()
        pre_demasked_index = (x != self.mask_id)
        memory_intervals = []

        steps_used = 0
        outputs = []
        confidences = []
        transfer_idxs = []
        history_intervals = []

        no_advance_n = 0
        for exp_step in range(exp_steps):
            x0, confidence, org_confidence, _ = self._model_forward(x, prompt_index)
            current_step += 1
            steps_used += 1
            GG = False

            # dynamic N
            unmasked_ratio = (x[:, self.block_start: self.block_end] != self.mask_id).sum().item() / self.block_length
            # select_N = max(1, round(self.exploration_N * (np.cos(np.pi/2 * unmasked_ratio))))
            select_N = self.exploration_N

            confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf  # semi support

            # mutiplying positional weights
            if self.positional_weights_type == 'absolute':
                confidence[:, self.block_start: self.block_end] = confidence[:, self.block_start: self.block_end] * self.absolute_positional_weights[current_step]
            elif self.positional_weights_type == 'ratio':
                # MARK
                global_unmasked_ratio = (x[:, prompt_len: ] != self.mask_id).sum().item() / self.gen_length
                dynamic_positional_weights = self.compute_dynamic_positional_weights(self.block_length, global_unmasked_ratio, device=x0.device)
                confidence[:, self.block_start: self.block_end] = confidence[:, self.block_start: self.block_end] * dynamic_positional_weights
            elif self.positional_weights_type == 'static':
                confidence[:, self.block_start: self.block_end] = confidence[:, self.block_start: self.block_end] * self.static_positional_weights[current_step]
            else:
                pass

            # seed tokens -> local clusters
            # 1. identify seed tokens
            conf_temp = confidence.clone()
            mask_index = (x == self.mask_id)
            conf_temp[~mask_index] = -np.inf
            select_N = min((conf_temp > 0).sum().item(), select_N)    # if no masked tokens, exits
            if select_N == 0:
                GG = True
            else:
                # applying Soft-NMS to select distant seed tokens
                # print(f"target select_N: {select_N}")
                seed_indices = [[] for _ in range(conf_temp.shape[0])]
                for b in range(conf_temp.shape[0]):
                    conf_b = conf_temp[b].clone()
                    conf_b.type(torch.float32)  # to avoid numeric underflow
                    idxs = torch.arange(conf_temp.shape[1], device=conf_temp.device, dtype=torch.float32)
                    # select a token with D(·) suppression, iteratively
                    for i in range(select_N):
                        select_idx = torch.argmax(conf_b).item()
                        # print(f"==> select_idx {select_idx}, conf: {conf_b[select_idx].item()}, token_id: {x[b, select_idx].item()}")
                        if conf_b[select_idx] <= 0 or conf_temp[b, select_idx] < self.exploration_threshold:
                            select_N = i
                            break
                        seed_indices[b].append(select_idx)  # selection
                        distances = torch.abs(idxs - select_idx)  # suppression
                        sigma = (conf_b.shape[0] - prompt_len) / select_N / 2
                        gauss_weights = 1 - torch.exp(-torch.pow(distances, 2) / (sigma ** 2))
                        conf_b = conf_b * gauss_weights
                        # print(f"conf_b after soft-NMS: {conf_b[prompt_len:].cpu().numpy()}")
                if select_N == 0:
                    GG = True
                else:
                    seed_indices = torch.Tensor(seed_indices).type(torch.long).to(x.device)

            if not GG:
                # 2. form clusters
                intervals = []
                exist_new_seed = False
                found_indices_in_memory = set()
                for b in range(seed_indices.shape[0]):
                    # determine from where to expansion: seed or previous interval
                    for interval_idx in range(select_N):
                        seed = seed_indices[b, interval_idx].item()
                        seed_conf = conf_temp[b, seed].item()

                        found_interval_in_memory = None
                        for idx, (start, end) in enumerate(memory_intervals):
                            if start <= seed <= end:
                                found_interval_in_memory = (start, end)  # (interval_idx, start, end)
                                found_indices_in_memory.add(idx)
                                break
                        if found_interval_in_memory is None:
                            assert x[b, seed] == self.mask_id
                            left, right = seed, seed
                            exist_new_seed = True
                        else:
                            left, right = found_interval_in_memory
                        # to left
                        n_consecutive_failures = 0
                        for i in range(left - 1, self.block_start - 1, -1):
                            # min(self.exploration_threshold, seed_conf)
                            if pre_demasked_index[b, i] or conf_temp[b, i] < self.exploration_threshold: # hit wall
                                n_consecutive_failures += 1
                                if n_consecutive_failures > self.exploration_M:
                                    break
                            else:
                                n_consecutive_failures = 0
                                left = i
                        # to right
                        n_consecutive_failures = 0
                        for i in range(right + 1, self.block_end):
                            if pre_demasked_index[b, i] or conf_temp[b, i] < self.exploration_threshold: # hit wall
                                n_consecutive_failures += 1
                                if n_consecutive_failures > self.exploration_M:
                                    break
                            else:
                                n_consecutive_failures = 0
                                right = i
                        # gathered: expanded interval in memory or new interval
                        intervals.append((left, right)) # [left, right]

                # merge intervals (gathered intervals + previous intervals in memory)
                old_intervals = [interval for idx, interval in enumerate(memory_intervals) if idx not in found_indices_in_memory]
                merged_intervals = self._merge_intervals(intervals + old_intervals)

                print(f"curr_step {current_step} [Exploration] (block-{self.num_block} unmasked_ratio: {unmasked_ratio:.2f}) exp_step {exp_step + 1}: select_N = {select_N}, found intervals {merged_intervals}")

                # transfer tokens: "gentle" parallel decoding: select_N tokens are updated
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                transfer_index.scatter_(dim=1, index=seed_indices, value=True)
                x[transfer_index] = x0[transfer_index]
                # x.scatter_(dim=1, src=x0, index=seed_indices) 天坑！！！ scatter方法的意思是从src中顺序地选择赋入x的index位置处。index指定的是x中要被赋值的位置而不是在src中要被选择用于赋值的位置。

                # condition check for early convergence
                advance_threshold = 0.1
                exist_intervals_advance = (
                        exp_step == 0 or
                        len(merged_intervals) != len(memory_intervals) or
                        any((merged_intervals[i][1] - merged_intervals[i][0] + 1) / (memory_intervals[i][1] - memory_intervals[i][0] + 1) >= 1 + advance_threshold for i in range(len(merged_intervals)))
                )

                memory_intervals = merged_intervals

                # for visualization
                outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
                confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
                transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])
                history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in merged_intervals])

                # if converged, early stop and exits the Divide phase
                if exist_new_seed or exist_intervals_advance:
                    no_advance_n = 0
                else:
                    no_advance_n += 1
                    # consecutive failures
                    if no_advance_n >= 2:
                        break
            else:
                print(f"curr_step {current_step} [Exploration] GG!!! (block-{self.num_block} unmasked_ratio: {unmasked_ratio:.2f}) exp_step {exp_step + 1}: select_N = {select_N}, with intervals {memory_intervals}")
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
                confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
                transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])
                history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in memory_intervals])

            self.exp_N = select_N

            if GG:
                break

        # post processing: density-based expansion + interval purification
        # 1. density-based expansion
        # density_expansion_intervals = []  # [(new_left, new_right), ...]
        # for i, (start, end) in enumerate(memory_intervals):
        #     interval_width = end - start + 1
        #     edge_width = max(3, int(interval_width * 0.4))  # marginal ratio = 0.4
        #
        #     left_density = org_confidence[:, start: start + edge_width].nan_to_num(neginf=0).mean().item()  # use org_confidence for robustness
        #     left_delta = int(left_density * edge_width)
        #     left = start
        #     n_hit_wall = 0
        #     left_search_min = max(self.block_start, start - left_delta)
        #     for pos in range(start - 1, left_search_min - 1, -1):
        #         if x[0, pos] != self.mask_id:
        #             n_hit_wall += 1
        #             if n_hit_wall > self.exploration_M:
        #                 break
        #         else:
        #             n_hit_wall = 0
        #             left = pos
        #
        #     right_density = org_confidence[:, end - edge_width + 1: end + 1].nan_to_num(neginf=0).mean().item()
        #     right_delta = int(right_density * edge_width)
        #     right = end
        #     n_hit_wall = 0
        #     right_search_max = min(self.block_end - 1, end + right_delta)
        #     for pos in range(end + 1, right_search_max + 1):
        #         if x[0, pos] != self.mask_id:
        #             n_hit_wall += 1
        #             if n_hit_wall > self.exploration_M:
        #                 break
        #         else:
        #             n_hit_wall = 0
        #             right = pos
        #
        #     density_expansion_intervals.append((left, right))
        #
        # memory_intervals = self._merge_intervals(density_expansion_intervals)

        # 2. interval purification: rule out intervals such that have been fully demasked during exploration steps
        purified_intervals = []
        for start, end in memory_intervals:
            left, right = start, end
            # unmasked contraction
            while left <= right and x[0, left].item() != self.mask_id:
                left += 1
            while left <= right and x[0, right].item() != self.mask_id:
                right -= 1
            if left <= right:
                purified_intervals.append((left, right))
            # if (x[0, start:end + 1] == self.mask_id).any():
            #     purified_intervals.append((start, end))
        memory_intervals = purified_intervals

        print(f"=====> constructed decoding zones: {memory_intervals}")
        # for visualization
        history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in memory_intervals])

        return x, memory_intervals, steps_used, outputs, confidences, transfer_idxs, history_intervals

    def acceleration_phase(
            self,
            x: Tensor,
            prompt_index: Tensor,
            intervals: List[tuple[int, int]] = None,
            current_step: int = -1
    ):
        """
        Conquer Phase: do confidence-based parallel decoding on decoding zones.
            intervals: decoding zones from the Divide phase
        """
        steps_used = 0
        if not intervals:
            return x, steps_used, [], [], []

        prompt_len = prompt_index[0].sum().item()
        # interval_states = [{'coords': (start, end), 'status': 'active'} for start, end in intervals]

        outputs = []
        confidences = []
        transfer_idxs = []
        history_intervals = []

        # while any(s['status'] == 'active' for s in interval_states):
        while len(intervals) > 0:
            # only focus on decoding zones
            dynamic_accel_mask = torch.zeros_like(x, dtype=torch.bool)
            n_active_intervals = 0
            # for state in interval_states:
            #     if state['status'] == 'active':
            #         start, end = state['coords']
            #         dynamic_accel_mask[:, start: end + 1] = True
            #         n_active_intervals += 1
            for (start, end) in intervals:
                dynamic_accel_mask[:, start: end + 1] = True
                n_active_intervals += 1
            if n_active_intervals == 0:
                break

            x0, confidence, org_confidence, _ = self._model_forward(x, prompt_index)
            current_step += 1
            steps_used += 1

            # mutiplying positional weights
            # if self.positional_weights_type == 'absolute':
            #     confidence[:, self.block_start: self.block_end] = confidence[:, self.block_start: self.block_end] * self.absolute_positional_weights[current_step]
            # elif self.positional_weights_type == 'ratio':
            #     # MARK
            #     global_unmasked_ratio = (x[:, prompt_len: ] != self.mask_id).sum().item() / self.gen_length
            #     dynamic_positional_weights = self.compute_dynamic_positional_weights(self.block_length, global_unmasked_ratio, device=x0.device)
            #     confidence[:, self.block_start: self.block_end] = confidence[:, self.block_start: self.block_end] * dynamic_positional_weights
            # elif self.positional_weights_type == 'static':
            #     confidence[:, self.block_start: self.block_end] = confidence[:, self.block_start: self.block_end] * self.static_positional_weights[current_step]
            # else:
            #     pass

            confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf   # semi support
            transfer_index = confidence > 0.98   # extreme confidence updating is safe
            # confidence_in_active_zones = torch.where(dynamic_accel_mask, confidence, -np.inf)

            # do confidence-base parallel decoding
            # TODO: 考虑在不同区间内结合各自的密度进行更新，可以基于区间密度平均(背诵vs灵感)，也可以将密度作为置信值的一部分(熟就是熟)，或二者都(数学式)
            total_n_para_updated = 0
            total_n_cons_updated = 0
            for (itv_start, itv_end) in intervals:
                # if state['status'] != 'active':
                #     continue
                # itv_start, itv_end = state['coords']

                mask_in_curr_zone = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                mask_in_curr_zone[:, itv_start:itv_end] = True
                confidence_in_curr_zone = torch.zeros_like(x0, dtype=confidence.dtype, device=x0.device)
                confidence_in_curr_zone[:, itv_start: itv_end + 1] = confidence[:, itv_start: itv_end + 1]
                # strategy1: parallel decoding based on confidence
                if self.acceleration_parallel_method == 'fixed':  # meaningless for Divide and Conquer
                    para_transfer_index = (confidence_in_curr_zone > self.acceleration_threshold)
                elif self.acceleration_parallel_method == 'factor':
                    # TODO: 3.使用Fast-dLLM中的公式: (n + 1) * (1 - c_{n}) < f 来确定最大的可并行解码n
                    # 1. 对>min_threshold的位置按confidence排序; 3. 对这些满足条件的index形成transfer_inedx
                    para_transfer_index = torch.zeros_like(confidence_in_curr_zone, dtype=torch.bool, device=x0.device)
                    for b in range(confidence_in_curr_zone.shape[0]):
                        conf_b = confidence_in_curr_zone[b].clone()
                        cand_mask = (conf_b > 0)  # (L,)
                        # 根据cand_confs排序cand_idxs
                        cand_idxs = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)  # (n,)
                        cand_confs = conf_b[cand_mask]  # (n,)
                        sorted_order = torch.argsort(cand_confs, descending=True)
                        cand_idxs = cand_idxs[sorted_order]
                        cand_confs = cand_confs[sorted_order]
                        # 2. 从cand_confs最低conf处开始挨个试验可行的n，直到满足条件;
                        for conf_idx, conf in reversed(list(enumerate(cand_confs.tolist()))):
                            para_feasible_n = int(self.acceleration_factor / (1 - conf + 1e-6) - 1)
                            #  3. 若满足公式，则根据这些满足条件的index形成transfer_inedx
                            if para_feasible_n >= conf_idx + 1:
                                para_transfer_index.scatter_(dim=1, index=cand_idxs[:conf_idx + 1].unsqueeze(0), value=True)
                                break
                elif self.acceleration_parallel_method == 'entropy':
                    pass
                elif self.acceleration_parallel_method == 'margin':
                    pass
                n_para_updated = para_transfer_index.sum().item()
                transfer_index |= para_transfer_index
                total_n_para_updated += n_para_updated

            if total_n_para_updated == 0:
                # final dance
                cons_transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                _, topk_idxs = torch.topk(confidence, k=min(1, (confidence > self.acceleration_low_threshold).sum().item()), dim=-1)  # (k,)
                cons_transfer_index.scatter_(dim=1, index=topk_idxs, value=True)
                total_n_cons_updated = cons_transfer_index.sum().item()
                transfer_index |= cons_transfer_index
            total_n_updated = total_n_para_updated + total_n_cons_updated

            x[transfer_index] = x0[transfer_index]
            print(f"curr_step {current_step} [Acceleration] (block-{self.num_block}): "
                  f"total_n_updated({total_n_updated}) = total_n_para_updated({total_n_para_updated})) + total_n_cons_updated({total_n_cons_updated})");

            # TODO: 3.根据区间边缘的密度，在一定程度上再动态推进区间（惯性）
            # enrolling expansion
            enrolled_intervals = []
            n_hit_tolerance = self.exploration_M
            for i, itv in enumerate(intervals):
                # if state['status'] == 'active':
                # enroll left
                start, end = itv

                # potential enrolling expansion
                left = start
                left_search_min = intervals[i-1][1] + 1 if i > 0 else self.block_start
                n_hit_wall = 0
                for pos in range(start - 1, left_search_min - 1, -1):
                    if confidence[0, pos] <= self.acceleration_low_threshold:
                        n_hit_wall += 1
                        if n_hit_wall > n_hit_tolerance:
                            break
                    else:
                        n_hit_wall = 0
                        left = pos
                right = end
                right_search_max = intervals[i+1][0] - 1 if i < len(intervals) - 1 else self.block_end - 1
                n_hit_wall = 0
                for pos in range(right + 1, right_search_max):
                    if confidence[0, pos] <= self.acceleration_low_threshold:
                        n_hit_wall += 1
                        if n_hit_wall > n_hit_tolerance:
                            break
                    else:
                        n_hit_wall = 0
                        right = pos

                # unmasked contraction
                while left <= right and x[0, left].item() != self.mask_id:
                    left += 1
                while left <= right and x[0, right].item() != self.mask_id:
                    right -= 1

                if left <= right and (x[:, left: right + 1] == self.mask_id).any():
                   enrolled_intervals.append((left, right)) # keep active

            intervals = self._merge_intervals(enrolled_intervals)
            print(f"intervals after enrolling: {intervals}")

            # for visualization
            history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in intervals])
            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])

            # exit when its speed <= that in Divide phase
            if total_n_updated <= self.exp_N:
                break

        return x, steps_used, outputs, confidences, transfer_idxs, history_intervals

    def mop_up_phase(
            self,
            x: Tensor,
            prompt_index: Tensor,
            current_step: int = -1,
    ):
        """
            Finalize Phase: do margin-based parallel+top1 decoding on global context。
        """

        todo_steps = math.ceil((x[:, self.block_start: self.block_end] == self.mask_id).sum(dim=1).item() / self.mopup_speed)

        outputs = []
        confidences = []
        transfer_idxs = []
        steps_used = 0

        if todo_steps <= 0:
            return x, steps_used, outputs, confidences, transfer_idxs

        prompt_len = prompt_index[0].sum().item()
        num_masked = (x[:, self.block_start: self.block_end] == self.mask_id).sum().item()
        for i in range(todo_steps):
            x0, confidence, org_confidence, logits = self._model_forward(x, prompt_index)
            steps_used += 1

            # transfer_index = conf_transfer_index = confidence > 0.98   # extreme confidence updating is safe
            # n_conf_updated = conf_transfer_index.sum().item()

            top2_logits = torch.topk(logits, k=2, dim=-1).values  # (b, l, 2)
            top2_margins = top2_logits[..., 0] - top2_logits[..., 1]  # (b, l)
            top2_margins[:, 0: self.block_start] = top2_margins[:, self.block_end:] = 0  # semi support
            #print statistics
            block_top2_margins = top2_margins[:, self.block_start: self.block_end]
            print(f"==> margins.mean={block_top2_margins.mean().item():.2f}, std={block_top2_margins.std().item():.2f}, "
                  f"max={block_top2_margins.max().item():.2f}, min={block_top2_margins.min().item():.2f}")
            top2_margins[x != self.mask_id] = 0

            transfer_index_margin = (top2_margins > self.mopup_margin_threshold)

            # n_margin_updated = (transfer_index_margin & ~conf_transfer_index).sum().item()
            n_margin_updated = transfer_index_margin.sum().item()
            n_topk_updated = min(num_masked, max(0, self.mopup_speed - n_margin_updated))
            if n_margin_updated > 0:
                transfer_index = transfer_index_margin
            else:
                confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf  # semi support
                # _, topk_idxs = torch.topk(confidence, k=num_transfer_tokens[0, i], dim=1)  # (b, l)
                _, topk_idxs = torch.topk(confidence, k=n_topk_updated, dim=1)  # (b, l)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                transfer_index.scatter_(dim=1, index=topk_idxs, value=True)
            x[transfer_index] = x0[transfer_index]
            num_masked = (x[:, self.block_start: self.block_end] == self.mask_id).sum().item()

            print(f"curr_step {current_step + i + 1} [Mop-up] (block-{self.num_block} unmasked_ratio: {num_masked / self.block_length:.2f}): n_updated({transfer_index.sum().item()}) = n_margin_updated({n_margin_updated}) + n_topk_updated({n_topk_updated})")

            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])

            if num_masked == 0:
                break

        return x, steps_used, outputs, confidences, transfer_idxs

    @torch.no_grad()
    def generate(
        self,
        prompt,
        gen_length=256,
        max_steps=256,
        block_length=256,
        enable_metrics=True,
    ) -> GenerateOutput:
        """
        DiCo Controller
        """
        assert gen_length <= self.model_max_genlength, f"gen_length must <= model_max_genlength({self.model_max_genlength})"
        assert max_steps <= self.model_max_steps, f"steps must <= model_max_steps({self.model_max_steps})"

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert max_steps % num_blocks == 0
        block_steps = max_steps // num_blocks

        self.gen_length = gen_length
        self.max_steps = max_steps
        self.block_length = block_length
        self.block_steps = block_steps

        # initalize positional weights
        if self.positional_weights_type == 'absolute':
            self.absolute_positional_weights = self.precompute_absolute_positional_weights(
                max_steps=max_steps, gen_length=block_length, device=self.model.device, dtype=torch.float32
            )
        elif self.positional_weights_type == 'static':
            self.static_positional_weights = self.precompute_static_positional_weights(
                gen_length=block_length, device=self.model.device, dtype=torch.float32
            )
        elif self.positional_weights_type == 'ratio':
            pass
        else:
            pass

        outputs = []
        confidences = []
        transfer_idxs = []
        phase_states = []  # [{'phase':'exploration/acceleration/mopup', 'range': (start, end)}]
        history_intervals_all = []  # [{'inceptive_step': 0, 'history_intervals': [[(start, end), ...], [(start, end), ...], ...]}]
        accumulated_steps = 0

        start_time = time.perf_counter()

        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = (x != self.mask_id)
        prompt_len = prompt_index[0].sum().item()

        for num_block in range(num_blocks):
            self.block_start = prompt_len + num_block * block_length
            self.block_end = prompt_len + (num_block + 1) * block_length
            self.num_block = num_block

            block_step_i = 0
            for EA_idx in range(int(block_steps * self.mopup_gate_ratio)):

                # ① Divide
                x, intervals, exploration_steps, exploration_outputs, exploration_confidences, exploration_transfer_idxs, history_intervals \
                    = self.exploration_phase(
                    x,
                    prompt_index,
                    exp_steps=min(self.max_exploration_steps, max_steps - accumulated_steps),
                    current_step=accumulated_steps
                )

                # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Exploration")
                outputs.extend(exploration_outputs)
                confidences.extend(exploration_confidences)
                transfer_idxs.extend(exploration_transfer_idxs)
                phase_states.append(
                    {'phase': 'exploration', 'range': (accumulated_steps, accumulated_steps + exploration_steps)})
                history_intervals_all.append({'inceptive_step': accumulated_steps, 'history_intervals': history_intervals})
                # print(f"exploration phase ends, use steps: {exploration_steps}, TPS: {(num_masked - num_masked_exploration) / (exploration_steps)}")
                # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Exploration")
                block_step_i += exploration_steps
                accumulated_steps += exploration_steps

                # ② Conquer
                if intervals:
                    x, acceleration_steps, outputs_acceleration, confidences_acceleration, transfer_idxs_acceleration, history_intervals \
                        = self.acceleration_phase(
                            x,
                            prompt_index,
                            intervals=intervals,
                            current_step=accumulated_steps,
                    )

                    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Acceleration")
                    outputs.extend(outputs_acceleration)
                    confidences.extend(confidences_acceleration)
                    transfer_idxs.extend(transfer_idxs_acceleration)
                    phase_states.append(
                        {'phase': 'acceleration', 'range': (accumulated_steps, accumulated_steps + acceleration_steps)})
                    history_intervals_all.append(
                        {'inceptive_step': accumulated_steps, 'history_intervals': history_intervals})
                    block_step_i += acceleration_steps
                    accumulated_steps += acceleration_steps
                    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Acceleration")

                    # check mopup condition in the current block
                    masked_ratio = 1.0 * (x[:, self.block_start: self.block_end] == self.mask_id).sum().item() / block_length
                    if masked_ratio < (1 - self.mopup_gate_ratio):
                        print(
                            f"block {num_block}: E-A turn ends with unmased ratio: {(1 - masked_ratio) * 100}% (>{self.mopup_gate_ratio * 100}%)")
                        break
                else:
                    break

            # ③ Finalize
            x, mopup_steps, outputs_mopup, confidences_mopup, transfer_idxs_mopup \
                = self.mop_up_phase(
                    x,
                    prompt_index,
                    current_step=accumulated_steps,
                )

            if mopup_steps > 0:
                outputs.extend(outputs_mopup)
                confidences.extend(confidences_mopup)
                transfer_idxs.extend(transfer_idxs_mopup)
                # print(f"mop_up phase ends, use steps: {mopup_steps}, TPS: {(num_masked_acceleration - num_masked_mopup) / (mopup_steps)}")
                phase_states.append({'phase': 'mopup', 'range': (accumulated_steps, accumulated_steps + mopup_steps)})
            else:
                pass
                # print(f"No Need for mop_up phase")

            block_step_i += mopup_steps
            accumulated_steps += mopup_steps
            print(f"block {num_block} is decoded over in step {accumulated_steps}.")

        # compute metrics
        end_time = time.perf_counter()
        duration = end_time - start_time
        total_steps = accumulated_steps

        metrics = GenerationMetrics(
            use_seconds=duration,
            use_steps=total_steps,
            n_gen_tokens=gen_length,
            tokens_per_second=(gen_length / duration) if duration > 0 else 0,
            step_reduction_ratio=(gen_length / total_steps) if total_steps > 0 else 0
        )
        print(metrics)

        return GenerateOutput(
            out=x,
            outputs=outputs,
            confidences=confidences,
            transfer_idxs=transfer_idxs,
            phase_states=phase_states,
            history_intervals_all=history_intervals_all,
            metrics=metrics,
        )

def main():
    set_seed(1234)
    device = 'cuda:1'
    model_path = "../models/LLaDA-8B-Instruct"

    # 4-shot prompt
    # few_shot_filename = "../prompts/gsm8k_shot.txt"
    # with open(few_shot_filename, "r", encoding="utf-8") as f:
    #     prompts= f.readlines()[0:3]

    # base gsm8k prompt
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    prompts = gsm8k_dataset['test']['question'][0:3]

    # base humaneval prompt
    # humaneval_dataset = load_dataset('openai/openai_humaneval')
    # prompts = humaneval_dataset['test']['prompt'][:5]

    # prompts = [
    #     "你知道周杰伦吗",
    #     "请写一首关于春天的诗",
    #     "请用Python写一个冒泡排序算法",
    # ]
    # --- 使用类进行生成 ---
    config = MRSamplerConfig(
        cfg_scale=0.0,
        temperature=0.0,
        max_exploration_steps=7,
        exploration_N=3,
        exploration_M=1,
        exploration_threshold=0.15,
        acceleration_parallel_method='factor',
        acceleration_threshold=0.9,
        acceleration_low_threshold=0.6,
        acceleration_factor=1,
        mopup_gate_ratio=0.8,
        mopup_margin_threshold=3,
        max_mopup_steps=20,
        mopup_speed=1,
        positional_weights_type='ratio',
        max_weight=1.0,
        initial_min_weight=0.05,
        ur_factor=1
    )

    sampler = MRSampler.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )

    # max_steps = 256
    # block_length = 64
    max_steps = 256
    block_lengthes = [128]
    # exploration_thresholds = [0.15, 0.25, 0.4] # -> 0.25 is good for 'fixed', 'factor'
    exploration_thresholds = [0.25]

    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)
        tokenizer = sampler.tokenizer

        m = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        # input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        for block_length in block_lengthes:
            print('=' * 20 + f" block_length: {block_length} " + "=" * 20)
            for exp_tr in exploration_thresholds:
                sampler.exploration_threshold = exp_tr
                print('=' * 20 + f" exploration_threshold: {exp_tr} " + "=" * 20)
                OUT = sampler.generate(input_ids, gen_length=max_steps, max_steps=max_steps, block_length=block_length)
                out = OUT.out
                ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                print(f"Prompt_{i}'s answer: {ans}\n")


if __name__ == '__main__':
    main()