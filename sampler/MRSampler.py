from typing import List, Tuple, Dict, Any, Literal
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
    min_k: int = 2
    # Mop-up phase config
    mopup_gate_ratio: float = 0.9
    max_mopup_steps: int = 10
    mopup_speed: int = 2


class MRSampler(BaseSampler):
    """
        该类封装了模型、分词器、所有超参数以及生成过程的三个阶段：
        1. 探索 (Exploration): 识别并建立高置信度的生成区间。
        2. 加速 (Acceleration): 在已识别的区间内进行高效、并行的解码。
        3. 收尾 (Mop-up): 用传统的自回归方法填充剩余的零散部分。
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
        一个辅助函数，用于合并重叠或相邻的区间。
        例如：[(10, 20), (15, 25), (40, 50)] -> [(10, 25), (40, 50)]
        """
        # 如果少于2个区间，则无需合并
        if len(intervals) < 2:
            return intervals

        # 1. 关键步骤：根据区间的起始点进行排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        # 初始化合并后列表
        merged_intervals = [sorted_intervals[0]]
        # 2. 遍历排序后的剩余区间
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
            block_mask: Tensor = None,
            exp_steps: int = 0,
            current_step: int = -1,
            EA_idx: int = 0
    ):
        """
        探索阶段：寻找多个高置信度区间
        exploration_N: 要寻找的区间数量
        exploration_M: 扩张时允许的最大连续失败次数
        threshold: 置信度阈值
        """
        # print("--- 进入exploration阶段 ---")
        memory_intervals = []  # 之前轮探索得到的intervals [(start1, end1), (start2, end2),...]
        prompt_len = prompt_index.sum(dim=1).item()
        gen_length = x.shape[-1] - prompt_len
        pre_demasked_index = (x != self.mask_id)

        steps_used = 0
        outputs = []
        confidences = []
        transfer_idxs = []
        history_intervals = []

        no_advance_n = 0
        for exp_step in range(exp_steps):

            x0, confidence, org_confidence = self._model_forward(x, prompt_index)
            current_step += 1
            steps_used += 1
            GG = False

            # dynamic N
            unmasked_ratio = (x[:, self.block_start: self.block_end] != self.mask_id).sum().item() / self.block_length
            # select_N = round((1 - unmasked_ratio) * (self.exploration_N - 1) + 1)
            select_N = max(1, round(self.exploration_N * (np.cos(np.pi/2 * unmasked_ratio))))

            confidence[~block_mask] = -np.inf

            # mutiplying positional weights
            if self.positional_weights_type == 'absolute':
                confidence[:, prompt_len:] = confidence[:, prompt_len:] * self.absolute_positional_weights[current_step]
            elif self.positional_weights_type == 'ratio':
                dynamic_positional_weights = self.compute_dynamic_positional_weights(gen_length, unmasked_ratio, device=x0.device)
                confidence[:, prompt_len:] = confidence[:, prompt_len:] * dynamic_positional_weights
            elif self.positional_weights_type == 'static':
                confidence[:, prompt_len:] = confidence[:, prompt_len:] * self.static_positional_weights[current_step]
            else:
                pass

            # 2. 寻找、扩张、合并区间
            # 2.1 寻找种子点
            conf_temp = confidence.clone()
            invalid_mask = (x != self.mask_id)
            # 在前几个探索阶段(EA_idx < 2)，禁止在前半段输出EOT符
            # if EA_idx < 2:
            #     mid_pred_point = prompt_len + (x.shape[1] - prompt_len) // 2
            #     invalid_mask |= ((torch.arange(x0.shape[1], device=x0.device) < mid_pred_point).unsqueeze(0)) & ((x0 == self.eot_id) | (x0 == self.endoftext_id))
            conf_temp[invalid_mask] = -np.inf
            select_N = min((~invalid_mask).sum().item(), select_N)    # mask且结果不是在前半段的EOT的位置数量
            if select_N == 0:
                GG = True

            if not GG:
                # soft-NMS式选seed点
                print(f"NO GG! select_N: {select_N}")
                seed_indices = [[] for _ in range(conf_temp.shape[0])]
                for b in range(conf_temp.shape[0]):
                    conf_b = conf_temp[b].clone()
                    conf_b.type(torch.float32)  # to avoid numeric underflow
                    idxs = torch.arange(conf_b.shape[0], device=conf_b.device, dtype=torch.float32)
                    # 使用基于高斯函数的soft-NMS来尽可能选到距离远的点
                    for i in range(select_N):
                        select_idx = torch.argmax(conf_b).item()
                        # 无点可选提前退出（如N=2，且这两个点距离<exploration_M）
                        if conf_b[select_idx] <= 0:
                            select_N = i
                            break
                        # 选择经抑制处理后置信度次高的点
                        seed_indices[b].append(select_idx)
                        # 使用高斯函数抑制近距离权重，使得下次倾向于选择远离当前位置的点
                        distances = torch.abs(idxs - select_idx)
                        sigma = (conf_b.shape[0] - prompt_len) / select_N / 2
                        gauss_weights = 1 - torch.exp(-torch.pow(distances, 2) / (sigma ** 2))
                        conf_b = conf_b * gauss_weights
                        # print(f"conf_b after soft-NMS: {conf_b[prompt_len:].cpu().numpy()}")
                seed_indices = torch.Tensor(seed_indices).type(torch.long).to(x.device)

                # 2.2 扩张 (方法1: 尽可能寻找多的区间)
                intervals = []

                exist_new_seed = False
                found_indices_in_memory = set()
                for b in range(seed_indices.shape[0]):
                    # 对于每个seed点，若其已在之前的某个区间内，则直接从当前区间的两侧探索，否则从它本身开始探索
                    for interval_idx in range(select_N):
                        seed = seed_indices[b, interval_idx].item()
                        seed_conf = conf_temp[b, seed].item()
                        left = -1
                        right = -1
                        # 寻找当前seed点是否已在区间内，从而确定本轮的探索起点
                        found_interval_in_memory = None
                        for idx, (start, end) in enumerate(memory_intervals):
                            if start <= seed <= end:
                                found_interval_in_memory = (start, end)  # (区间编号, 起始, 结束)
                                found_indices_in_memory.add(idx)
                                break
                        if found_interval_in_memory is None:
                            assert x[0, seed] == self.mask_id
                            left, right = seed, seed
                            exist_new_seed = True
                        else:
                            left, right = found_interval_in_memory
                        # 向左扩张...
                        n_consecutive_failures = 0
                        for i in range(left - 1, prompt_len - 1, -1):
                            if pre_demasked_index[b, i] or conf_temp[b, i] < min(self.exploration_threshold, seed_conf): #碰到demask已被demask的位置停止
                                n_consecutive_failures += 1
                                if n_consecutive_failures > self.exploration_M:
                                    break
                            else:
                                n_consecutive_failures = 0
                                left = i
                        # left += pre_demasked_index[b, left: left + self.exploration_M].sum().item()  # re-pull
                        # 向右扩张...
                        n_consecutive_failures = 0
                        for i in range(right + 1, x.shape[1]):
                            if pre_demasked_index[b, i] or conf_temp[b, i] < min(self.exploration_threshold, seed_conf): #碰到demask已被demask的位置停止
                                n_consecutive_failures += 1
                                if n_consecutive_failures > self.exploration_M:
                                    break
                            else:
                                n_consecutive_failures = 0
                                right = i
                        # right -= pre_demasked_index[b, right - self.exploration_M + 1: right + 1].sum().item()  # re-pull
                        # 本轮收集到区间: memory中原对应区间的拓展 + 本轮探索新开区间
                        intervals.append((left, right)) # 左右双闭 [left, right]

                # 2.3 合并 (收集到区间 + memory中未在本轮中继续拓展的区间)
                old_intervals = [interval for idx, interval in enumerate(memory_intervals) if idx not in found_indices_in_memory]
                merged_intervals = self._merge_intervals(intervals + old_intervals)

                # 3.top-k填充
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                transfer_index.scatter_(dim=1, index=seed_indices, value=True)
                x[transfer_index] = x0[transfer_index]
                # x.scatter_(dim=1, src=x0, index=seed_indices) 天坑！！！ scatter方法的意思是从src中顺序地选择赋入x的index位置处。index指定的是x中要被赋值的位置而不是在src中要被选择用于赋值的位置。

                print(f"curr_step {current_step} [Exploration] (unmasked_ratio: {unmasked_ratio:.2f}) exp_step {exp_step + 1}: select_N = {select_N}, found intervals {merged_intervals}")

                # 4. 历史区间更新
                advance_threshold = 0.1
                exist_intervals_advance = (
                        exp_step == 0 or
                        len(merged_intervals) != len(memory_intervals) or
                        any((merged_intervals[i][1] - merged_intervals[i][0] + 1) / (memory_intervals[i][1] - memory_intervals[i][0] + 1) >= 1 + advance_threshold for i in range(len(merged_intervals)))
                )

                memory_intervals = merged_intervals

                # 研究用：中间状态收集
                outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
                confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
                transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])
                history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in merged_intervals])

                # 5. 探索阶段提前结束判断
                if exist_new_seed or exist_intervals_advance:
                    no_advance_n = 0
                else:
                    no_advance_n += 1
                    # 后续可改为区间熵不变
                    if no_advance_n >= 2:
                        break
            else:
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
                confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
                transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])
                history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in memory_intervals])

            self.exp_N = select_N   #ddd

            if GG:
                break

        # post processing
        # 1. density-based expansion
        # print(f"探索阶段找到原始区间: {memory_intervals}")
        # edp_ratio = 2 * (1.0) / select_N
        # desities = [org_confidence[:, start: end+1].nan_to_num(neginf=0).mean().item() for start, end in memory_intervals]
        # expected_deltas = [int(desities[i] * (end - start + 1) * edp_ratio) for i, (start, end) in enumerate(memory_intervals)]
        density_expansion_intervals = []  # [(new_left, new_right), ...]
        for i, (start, end) in enumerate(memory_intervals):
            # 两边总扩最大delta个格子，尽可能均衡拓展
            # delta = expected_deltas[i]
            interval_width = end - start + 1
            # 计算边缘区域的宽度，至少为1
            edge_width = max(3, int(interval_width * 0.4))

            left_density = org_confidence[:, start: start + edge_width].nan_to_num(neginf=0).mean().item()
            left_delta = int(left_density * edge_width)
            left = start
            n_hit_wall = 0
            left_search_min = max(prompt_len, start - left_delta)
            for pos in range(start - 1, left_search_min - 1, -1):
                if x[0, pos] != self.mask_id:
                    n_hit_wall += 1
                    if n_hit_wall > self.exploration_M:
                        break
                else:
                    n_hit_wall = 0
                    left = pos

            right_density = org_confidence[:, end - edge_width + 1: end + 1].nan_to_num(neginf=0).mean().item()
            right_delta = int(right_density * edge_width)
            right = end
            n_hit_wall = 0
            right_search_max = min(x.shape[1] - 1, end + right_delta)
            for pos in range(end + 1, right_search_max + 1):
                if x[0, pos] != self.mask_id:
                    n_hit_wall += 1
                    if n_hit_wall > self.exploration_M:
                        break
                else:
                    n_hit_wall = 0
                    right = pos

            density_expansion_intervals.append((left, right))

            # # 左侧搜索范围
            # left_search_min = max(prompt_len, start - math.ceil(delta / 2))
            # left = start
            # n_hit_wall = 0
            # for pos in range(start - 1, left_search_min - 1, -1):
            #     if x[0, pos] != self.mask_id:
            #         n_hit_wall += 1
            #         if n_hit_wall > self.exploration_M:
            #             break
            #     else:
            #         n_hit_wall = 0
            #         left = pos
            # # 右侧搜索范围
            # rest_delta = delta - (start - left)
            # right_search_max = min(x.shape[1] - 1, end + rest_delta)
            # right = end
            # n_hit_wall = 0
            # for pos in range(end + 1, right_search_max + 1):
            #     if x[0, pos] != self.mask_id:
            #         n_hit_wall += 1
            #         if n_hit_wall > self.exploration_M:
            #             break
            #     else:
            #         n_hit_wall = 0
            #         right = pos
            # # 补左搜索(左扩已==delta/2，右扩<delta/2<=rest_delta，剩余格再分配到左)
            # rest_delta = rest_delta - (right - end)
            # left_search_min = max(prompt_len, left - rest_delta)
            # n_hit_wall = 0
            # for pos in range(left - 1, left_search_min - 1, -1):
            #     if x[0, pos] != self.mask_id:
            #         n_hit_wall += 1
            #         if n_hit_wall > self.exploration_M:
            #             break
            #     else:
            #         n_hit_wall = 0
            #         left = pos
            # density_expansion_intervals.append((left, right))
        memory_intervals = self._merge_intervals(density_expansion_intervals)

        # 2. interval purification: rule out intervals such that have been fully demasked during exploration steps
        purified_intervals = []
        for start, end in memory_intervals:
            if (x[0, start:end + 1] == self.mask_id).any():
                purified_intervals.append((start, end))
        memory_intervals = purified_intervals

        print(f"边界扩张+区间净化后形成区间: {memory_intervals}")
        history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in memory_intervals])    # for visualization

        return x, memory_intervals, steps_used, outputs, confidences, transfer_idxs, history_intervals

    def acceleration_phase(
            self,
            x: Tensor,
            prompt_index: Tensor,
            block_mask: Tensor = None,
            intervals: List[tuple[int, int]] = None,
            current_step: int = -1
    ):
        """
        自适应加速阶段: 对探索阶段中形成的可行区间进行高速解码，直至不符合某种下限条件
            每个区间都有独立的生命周期。
            acceleration_method: factor/threshold
        """
        # print(f"开始加速区间: {intervals}")
        steps_used = 0
        if not intervals:
            return x, steps_used, [], [], []

        # 初始化状态
        prompt_len = prompt_index.sum(dim=1).item()
        gen_length = x.shape[-1] - prompt_len
        interval_states = [{'coords': (start, end), 'status': 'active'} for start, end in intervals]

        # 状态收集，用于可视化
        outputs = []
        confidences = []
        transfer_idxs = []

        # 只要还存在活着的区间，就继续
        while any(s['status'] == 'active' for s in interval_states):
            # 动态创建只包含当前要加速的区间的mask
            dynamic_accel_mask = torch.zeros_like(x, dtype=torch.bool)
            n_active_intervals = 0
            for state in interval_states:
                if state['status'] == 'active':
                    start, end = state['coords']
                    dynamic_accel_mask[:, start: end + 1] = True
                    n_active_intervals += 1
            if n_active_intervals == 0:
                break

            # 执行一次forward并行更新
            x0, confidence, org_confidence = self._model_forward(x, prompt_index)
            current_step += 1
            steps_used += 1
            GG = False

            # semi support
            # confidence[~block_mask] = -np.inf
            confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf

            # mutiplying positional weights
            # if self.positional_weights_type == 'absolute':
            #     confidence[:, prompt_len:] = confidence[:, prompt_len:] * self.absolute_positional_weights[current_step]
            # elif self.positional_weights_type == 'ratio':
            #     unmasked_ratio = (x[:, self.block_start: self.block_end] != self.mask_id).sum().item() / self.block_length
            #     dynamic_positional_weights = self.compute_dynamic_positional_weights(gen_length, unmasked_ratio, device=x0.device)
            #     confidence[:, prompt_len:] = confidence[:, prompt_len:] * dynamic_positional_weights
            # elif self.positional_weights_type == 'static':
            #     confidence[:, prompt_len:] = confidence[:, prompt_len:] * self.static_positional_weights[current_step]
            # else:
            #     pass

            confidence_in_active_zones = torch.where(dynamic_accel_mask, confidence, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            # do 2-strategies updating
            # TODO: 考虑在不同区间内结合各自的密度进行更新，可以基于区间密度平均(背诵vs灵感)，也可以将密度作为置信值的一部分(熟就是熟)，或二者都(数学式)
            total_n_para_updated = 0
            total_n_cons_updated = 0
            for state in interval_states:
                if state['status'] != 'active':
                    continue

                itv_start, itv_end = state['coords']
                mask_in_curr_zone = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                mask_in_curr_zone[:, itv_start:itv_end] = True
                confidence_in_curr_zone = torch.zeros_like(x0, dtype=confidence_in_active_zones.dtype, device=x0.device)
                confidence_in_curr_zone[:, itv_start: itv_end + 1] = confidence_in_active_zones[:, itv_start: itv_end + 1]
                # 更新策略1[加速]: 并行解码策略更新
                if self.acceleration_parallel_method == 'fixed':
                    para_transfer_index = (confidence_in_curr_zone > self.acceleration_threshold)
                elif self.acceleration_parallel_method == 'factor':
                    # TODO: 3.使用Fast-dLLM中的公式: (n + 1) * (1 - c_{n}) < f 来确定最大的可并行解码n
                    # 1. 对>min_threshold的位置按confidence排序; 3. 对这些满足条件的index形成transfer_inedx
                    cand_threshold = 0.0 * self.acceleration_threshold + 1.0 * self.acceleration_low_threshold
                    cand_mask = (confidence_in_curr_zone > cand_threshold)  # (L,)
                    # print(f"LB-satisfied #tokens: {cand_mask.sum().item()}")
                    cand_idxs = torch.nonzero(cand_mask, as_tuple=False)[:, 1]  # (n,)
                    cand_confs = confidence_in_curr_zone[cand_mask]  # (n,)
                    # 根据cand_confs排序cand_idxs
                    sorted_order = torch.argsort(cand_confs, descending=True)
                    cand_idxs = cand_idxs[sorted_order]
                    cand_confs = cand_confs[sorted_order]
                    # 2. 从cand_confs最低conf处开始挨个试验可行的n，直到满足条件;
                    for conf_idx, conf in reversed(list(enumerate(cand_confs.tolist()))):
                        para_feasible_n = int(self.acceleration_factor / (1 - conf + 1e-6) - 1)
                        #  3. 若满足公式，则根据这些满足条件的index形成transfer_inedx
                        if para_feasible_n >= conf_idx + 1:
                            transfer_index.scatter_(dim=1, index=cand_idxs[:conf_idx + 1].unsqueeze(0), value=True)
                            break
                elif self.acceleration_parallel_method == 'entropy':
                    pass
                elif self.acceleration_parallel_method == 'margin':
                    pass
                n_para_updated = para_transfer_index.sum().item()
                # 更新策略2[保底，暂时取消]: 若区间内元素不达解码阈值但仍高于最低阈值(可以直接是探索区间的阈值), 则进行top-k保守(conservative)更新
                # n_cons_updated = 0
                # cons_transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                # cons_min_k = max(1, self.exp_N // n_active_intervals)
                # if n_para_updated < cons_min_k:
                #     confidence_for_topk = confidence_in_curr_zone.clone()
                #     # 排除已经为True的位置
                #     if n_para_updated > 0:
                #         confidence_for_topk[para_transfer_index] = -np.inf
                #     _, topk_index = torch.topk(confidence_for_topk, dim=1, k=cons_min_k - n_para_updated)
                #     # 最低置信度检测
                #     cons_mask = confidence_for_topk.gather(dim=1, index=topk_index) > self.acceleration_low_threshold
                #     cons_index = topk_index[cons_mask].unsqueeze(0)  # topk_index[topk_mask]会将结果自动降维
                #     n_cons_updated = cons_mask.sum().item()
                #     # 增加填充位置
                #     cons_transfer_index.scatter_(dim=1, index=cons_index, value=True)
                # 为transfer_index增加当前区间的更新信息
                transfer_index |= para_transfer_index
                total_n_para_updated += n_para_updated
                # transfer_index |= cons_transfer_index
                # total_n_updated += n_cons_updated

            total_n_updated = total_n_para_updated + total_n_cons_updated
            if total_n_updated > 0:
                x[transfer_index] = x0[transfer_index]
                # print(f"transfered positions: {torch.where(transfer_index)[-1]}")
                # TODO: 3.根据区间边缘的密度，在一定程度上再动态推进区间（惯性）
                # 更新每个活跃区间的状态
                for state in interval_states:
                    if state['status'] == 'active':
                        start, end = state['coords']
                        # 检查这个区域是否还有[MASK]
                        if not (x[:, start: end + 1] == self.mask_id).any():
                            state['status'] = 'completed'
                            # print(f"区间 {state['coords']} 加速完成!")

            print(f"curr_step {current_step} [Acceleration]: "
                  f"total_n_updated({total_n_updated}) = total_n_para_updated({total_n_para_updated})) + total_n_cons_updated({total_n_cons_updated})");

            # dev: 更新历史记录
            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])

            # 单轮更新数低于设置阈值，提前退出加速阶段
            if total_n_updated < self.exp_N:
                break

        return x, steps_used, outputs, confidences, transfer_idxs

    def mop_up_phase(
            self,
            x: Tensor,
            prompt_index: Tensor,
            block_mask: Tensor = None,
            current_step: int = -1,
            block_step_i: int = -1
    ):
        """
            阶段三：收尾阶段
            用传统的top-k方式，填充所有剩余的[MASK]位置。
            speed: 每次decode的[MASK]数量，默认为2
        """

        todo_steps = math.ceil((x[:, self.block_start: self.block_end] == self.mask_id).sum(dim=1).item() / self.mopup_speed)
        todo_steps = min(todo_steps, self.block_steps - block_step_i)

        outputs = []
        confidences = []
        transfer_idxs = []
        steps_used = 0

        if todo_steps <= 0:
            # print("Mopup phase: all tokens decoded, skip mop_up phase.")
            return x, steps_used, outputs, confidences, transfer_idxs

        prompt_len = prompt_index[0].sum().item()
        mask_index = (x == self.mask_id)
        mask_index[~block_mask] = False     #semi-ar support
        num_transfer_tokens = get_num_transfer_tokens(mask_index, todo_steps)  # (b, todo_steps)

        for i in range(todo_steps):
            x0, confidence, org_confidence = self._model_forward(x, prompt_index)
            steps_used += 1

            confidence[~block_mask] = -np.inf   #semi support

            _, topk_idxs = torch.topk(confidence, k=num_transfer_tokens[0, i], dim=1)  # (b, l)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index.scatter_(dim=1, index=topk_idxs, value=True)
            x[transfer_index] = x0[transfer_index]

            print(f"curr_step {current_step + i + 1} [Mop-up]: n_transferred({transfer_index.sum().item()})")
            # x.scatter_(dim=1, index=topk_idxs, src=x0) scatter天坑！理由见exploration中的scatter使用

            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])

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
        实现“多区域并行置信度驱动解码”思路的主函数。
        """
        # 初始化

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
                max_steps=max_steps, gen_length=gen_length, device=self.model.device, dtype=torch.float32
            )
        elif self.positional_weights_type == 'static':
            self.static_positional_weights = self.precompute_static_positional_weights(
                gen_length=gen_length, device=self.model.device, dtype=torch.float32
            )
        elif self.positional_weights_type == 'ratio':
            pass
        else:
            pass

        # 主循环 (探索与加速)
        outputs = []
        confidences = []
        transfer_idxs = []
        phase_states = []  # [{'phase':'exploration/acceleration/mopup', 'range': (start, end)}]
        exploration_intervals = []  # [{'inceptive_step': 0, 'history_intervals': [[(start, end), ...], [(start, end), ...], ...]}]
        accumulated_steps = 0

        # print(f"Starting Inference ============================= {x.shape}")
        start_time = time.perf_counter()

        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = (x != self.mask_id)
        prompt_len = prompt_index.sum(dim=1).item()

        for num_block in range(num_blocks):
            self.block_start = prompt_len + num_block * block_length
            self.block_end = prompt_len + (num_block + 1) * block_length

            block_mask = torch.ones((1, prompt_len + gen_length), dtype=torch.bool).to(self.device)
            block_mask[:, prompt_len + (num_block + 1) * block_length:] = 0

            block_step_i = 0
            for EA_idx in range(block_steps - self.max_mopup_steps):
                # check mopup condition in the current block
                num_masked = (x[block_mask] == self.mask_id).sum().item()
                masked_ratio = 1.0 * num_masked / block_length
                if masked_ratio < (1 - self.mopup_gate_ratio):
                    print(f"block {num_block}: E-A turn ends with unmased ratio: {(1 - masked_ratio) * 100}% (>{self.mopup_gate_ratio * 100}%)")
                    break

                # ① Divide
                x, intervals, exploration_steps, exploration_outputs, exploration_confidences, exploration_transfer_idxs, history_intervals \
                    = self.exploration_phase(
                    x,
                    prompt_index,
                    block_mask=block_mask,
                    exp_steps=min(self.max_exploration_steps, max_steps - accumulated_steps),
                    EA_idx=EA_idx,
                    current_step=accumulated_steps
                )

                # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Exploration")
                outputs.extend(exploration_outputs)
                confidences.extend(exploration_confidences)
                transfer_idxs.extend(exploration_transfer_idxs)
                phase_states.append(
                    {'phase': 'exploration', 'range': (accumulated_steps, accumulated_steps + exploration_steps)})
                exploration_intervals.append({'inceptive_step': accumulated_steps, 'history_intervals': history_intervals})
                # print(f"exploration phase ends, use steps: {exploration_steps}, TPS: {(num_masked - num_masked_exploration) / (exploration_steps)}")
                # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Exploration")
                block_step_i += exploration_steps
                accumulated_steps += exploration_steps

                # ② Conquer
                if intervals:
                    x, acceleration_steps, outputs_acceleration, confidences_acceleration, transfer_idxs_acceleration \
                        = self.acceleration_phase(
                            x,
                            prompt_index,
                            block_mask=block_mask,
                            intervals=intervals,
                            current_step=accumulated_steps,
                    )

                    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Acceleration")
                    outputs.extend(outputs_acceleration)
                    confidences.extend(confidences_acceleration)
                    transfer_idxs.extend(transfer_idxs_acceleration)
                    phase_states.append(
                        {'phase': 'acceleration', 'range': (accumulated_steps, accumulated_steps + acceleration_steps)})
                    block_step_i += acceleration_steps
                    accumulated_steps += acceleration_steps
                    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Acceleration")

            # ③ Finalize
            # TODO: 根据当前自信度(第1和第2答案的logits均值)，决定进入收尾还是直接commit答案
            # print("\n--- 进入收尾阶段 ---")
            x, mopup_steps, outputs_mopup, confidences_mopup, transfer_idxs_mopup \
                = self.mop_up_phase(
                    x,
                    prompt_index,
                    block_mask=block_mask,
                    current_step=accumulated_steps,
                    block_step_i=block_step_i
                )
            # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Mopup")
            if mopup_steps > 0:
                outputs.extend(outputs_mopup)
                confidences.extend(confidences_mopup)
                transfer_idxs.extend(transfer_idxs_mopup)
                # print(f"mop_up phase ends, use steps: {mopup_steps}, TPS: {(num_masked_acceleration - num_masked_mopup) / (mopup_steps)}")
                phase_states.append({'phase': 'mopup', 'range': (accumulated_steps, accumulated_steps + mopup_steps)})
            else:
                pass
                # print(f"No Need for mop_up phase")
            # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Mopup")
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
            exploration_intervals=exploration_intervals,
            metrics=metrics,
        )

def main():
    set_seed(1234)
    device = 'cuda:0'
    model_path = "../models/LLaDA-8B-Instruct"

    # 4-shot prompt
    # few_shot_filename = "../prompts/gsm8k_shot.txt"
    # with open(few_shot_filename, "r", encoding="utf-8") as f:
    #     prompts= f.readlines()[0:3]

    # base gsm8k prompt
    # gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    # prompts = gsm8k_dataset['test']['question'][0:2]

    # base humaneval prompt
    humaneval_dataset = load_dataset('openai/openai_humaneval')
    prompts = humaneval_dataset['test']['prompt'][:5]

    # prompts = [
    #     "你知道周杰伦吗",
    #     "请写一首关于春天的诗",
    #     "请用Python写一个冒泡排序算法",
    # ]
    # --- 使用类进行生成 ---
    config = MRSamplerConfig(
        cfg_scale=0.0,
        temperature=0.0,
        max_exploration_steps=10,
        exploration_N=1,
        exploration_M=2,
        exploration_threshold=0.15,
        acceleration_parallel_method='fixed',
        acceleration_threshold=0.9,
        acceleration_low_threshold=0.6,
        acceleration_factor=1,
        max_mopup_steps=10,
        mopup_gate_ratio=0.80,
        mopup_speed=2,
        positional_weights_type='none',
        max_weight=1.0,
        initial_min_weight=0.0,
    )

    sampler = MRSampler.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )


    max_steps = 256
    block_length = 64
    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)
        tokenizer = sampler.tokenizer

        m = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

        # exploration_thresholds = [0.15, 0.25, 0.4] # -> 0.25 is good for 'fixed', 'factor'
        exploration_thresholds = [0.25]
        for exp_tr in exploration_thresholds:
            sampler.exploration_threshold = exp_tr
            print('=' * 20 + f" exploration_threshold: {exp_tr} " + "=" * 20)
            OUT = sampler.generate(input_ids, gen_length=max_steps, max_steps=max_steps, block_length=block_length)
            out = OUT.out
            ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            print(f"Prompt_{i}'s answer: {ans}\n")


if __name__ == '__main__':
    main()