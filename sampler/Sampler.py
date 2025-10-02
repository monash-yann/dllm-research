from typing import List, Tuple
from torch import Tensor

import torch
import time
import math
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

from sampler.utils import add_gumbel_noise, get_num_transfer_tokens


class MRSampler:
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
            # Generation general config
            model_max_genlength: int = 2048,
            model_max_steps: int = 2048,
            # Special token ids
            mask_id: int = 126336,
            endoftext_id: int = 126081,
            eot_id: int = 126348,
            # Model forward config
            cfg_scale: float = 0.0,
            temperature: float = 0.0,
            # Exploration phase config
            max_exploration_steps: int = 5,
            N: int = 2,
            M: int = 2,
            exploration_threshold: float = 0.2,
            # Acceleration phase config
            acceleration_threshold: float = 0.85,
            acceleration_low_threshold: float = 0.5,
            acceleration_factor: float = 1.0,
            min_k: int = 2,
            # Mop-up phase config
            mopup_gate_ratio: float = 0.9,
            max_mopup_steps: int = 20,
            mopup_speed: int = 2,
            # position-step weight encoding config
            position_weight_initial_min: float = 0.2,
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        # generating configs
        self.model_max_genlength = model_max_genlength
        self.model_max_steps = model_max_steps
        self.mask_id = mask_id
        self.endoftext_id = endoftext_id
        self.eot_id = eot_id
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        # hyperparameters
        self.max_exploration_steps = max_exploration_steps
        self.N = N
        self.M = M
        self.exploration_threshold = exploration_threshold
        self.mopup_gate_ratio = mopup_gate_ratio
        self.acceleration_threshold = acceleration_threshold
        self.acceleration_low_threshold = acceleration_low_threshold
        self.acceleration_factor = acceleration_factor
        self.min_k = min_k
        self.max_mopup_steps = max_mopup_steps
        self.mopup_speed = mopup_speed

    @classmethod
    def from_path(
            cls,
            model_path: str,
            device: str = 'cuda:0',
            torch_dtype: torch.dtype = torch.bfloat16,
            **kwargs
    ):
        print(f"Loading model and tokenizer from path: {model_path}")

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        return cls(model=model, tokenizer=tokenizer, **kwargs)

    def _precompute_positional_weights(
        self,
        max_steps: int,
        gen_length: int,
        max_weight: float,
        initial_min_weight: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
            precompute a weight matrix shaped (max_steps, gen_length)
            每一行代表一个step的权重曲线，曲线随step增加而变得平缓。
        """
        assert gen_length > 0 and max_steps > 0, "gen_length and max_steps must > 0"
        if gen_length == 1:
            return torch.full((max_steps, gen_length), max_weight, device=device, dtype=dtype)

        positions = torch.arange(gen_length, device=device, dtype=dtype).unsqueeze(0)  # (1, gen_length)
        if max_steps == 1:
            lambda_decay = -torch.log(torch.tensor(initial_min_weight, device=device, dtype=dtype)) / (gen_length - 1)
            return torch.exp(-lambda_decay * positions)

        # compute positional weights
        steps = torch.arange(max_steps, device=device, dtype=torch.float32).unsqueeze(1)  # (max_steps, 1)
        # compute min_weight on each step via linear interpolation
        t = steps / (max_steps - 1)  # interpolation factor
        min_weights = initial_min_weight + (max_weight - initial_min_weight) * t  # (max_steps, 1)
        # compute lambda_decay on each step, according to t
        lambda_decays = -torch.log(min_weights) / (gen_length - 1)  # (max_steps, 1)
        # compute step_position_weights via broadcasting
        step_position_weights = torch.exp(-lambda_decays * positions)  # (max_steps, gen_length)

        return step_position_weights


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

    @torch.no_grad()
    def _model_forward(
            self,
            x: torch.Tensor,
            prompt_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return x0, confidence


    #  core methods  #
    def exploration_phase(
            self,
            x: Tensor,
            prompt_index: Tensor,
            position_weights: Tensor,
            EA_idx: int = 0,
            current_step: int = 0
    ):
        """
        探索阶段：寻找多个高置信度区间
        N: 要寻找的区间数量
        M: 扩张时允许的最大连续失败次数
        threshold: 置信度阈值
        """

        # print("--- 进入exploration阶段 ---")
        memory_intervals = []  # 之前轮探索得到的intervals [(start1, end1), (start2, end2),...]
        prompt_len = prompt_index.sum(dim=1).item()
        pre_demasked_index = (x != self.mask_id)

        steps_used = 0
        outputs = []
        confidences = []
        transfer_idxs = []
        history_intervals = []

        consecutive_sameness = 0
        for exp_step in range(self.max_exploration_steps):
            x0, confidence = self._model_forward(x, prompt_index)
            # inserting positional weights
            confidence[:, prompt_len:] = confidence[:, prompt_len:] * position_weights[current_step]
            # 2. 寻找、扩张、合并区间 (这里的逻辑来自我们上一轮的实现)
            # 2.1 寻找种子点
            conf_temp = confidence.clone()
            invalid_mask = (x != self.mask_id)
            # 在前几个探索阶段(EA_idx < 2)，禁止在前半段输出EOT符
            if EA_idx < 2:
                mid_pred_point = prompt_len + (x.shape[1] - prompt_len) // 2
                invalid_mask |= ((torch.arange(x0.shape[1], device=x0.device) < mid_pred_point).unsqueeze(0)) & ((x0 == self.eot_id) | (x0 == self.endoftext_id))
            conf_temp[invalid_mask] = -np.inf
            select_N = min((~invalid_mask).sum().item(), self.N)    # mask且结果不是在前半段的EOT的位置数量
            if select_N == 0:
                break

            _, seed_indices = torch.topk(conf_temp, k=select_N, dim=1)      # (b, l)

            # NMS抑制所有种子点选到同一区间
            seed_indices = [[]]
            for b in range(conf_temp.shape[0]):
                conf_b = conf_temp[b].clone()  # 创建一个副本以进行修改
                # 进行N轮NMS抑制选择
                for i in range(select_N):
                    select_idx = torch.argmax(conf_b).item()
                    # 无点可选提前退出（如N=2，且这两个点距离<M）
                    if conf_b[select_idx] <= -np.inf:
                        select_N = i
                        break
                    # 选择经抑制处理后置信度次高的点
                    seed_indices[b].append(select_idx)
                    # 若该点是孤立点，则抑制该点及其周围 M 范围内的邻居；若该点已在某个区间内，则抑制整个区间
                    found_interval_in_memory = None
                    for idx, (start, end) in enumerate(memory_intervals):
                        if start <= select_idx <= end:
                            found_interval_in_memory = (start, end)  # (区间编号, 起始, 结束)
                            break
                    if found_interval_in_memory is not None:
                        nms_start = max(0, found_interval_in_memory[0] - self.M)
                        nms_end = min(conf_b.shape[0], found_interval_in_memory[1] + self.M)
                    else:
                        nms_start = max(0, select_idx - self.M)
                        nms_end = min(conf_b.shape[0], select_idx + self.M)
                    conf_b[nms_start: nms_end+1] = -np.inf
            seed_indices = torch.Tensor(seed_indices).type(torch.long).to(x.device)

            # 2.2 扩张 (方法1: 尽可能寻找多的区间)
            intervals = []

            exist_new_seed = False
            found_indices_in_memory = set()
            for b in range(seed_indices.shape[0]):
                # 对于每个seed点，若其已在之前的某个区间内，则直接从当前区间的两侧探索，否则从它本身开始探索
                for interval_idx in range(select_N):
                    seed = seed_indices[b, interval_idx].item()
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
                    consecutive_failures = 0
                    for i in range(left - 1, prompt_len - 1, -1):
                        if pre_demasked_index[b, i]: #碰到demask已被demask的位置停止
                            break
                        if conf_temp[b, i] > self.exploration_threshold:
                            left = i
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            if consecutive_failures > self.M:
                                break
                    # 向右扩张...
                    consecutive_failures = 0
                    for i in range(right + 1, x.shape[1]):
                        if pre_demasked_index[b, i]: #碰到demask已被demask的位置停止
                            break
                        if conf_temp[b, i] > self.exploration_threshold:
                            right = i
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            if consecutive_failures > self.M:
                                break
                    # 本轮收集到区间: memory中原对应区间的拓展 + 本轮探索新开区间
                    intervals.append((left, right)) # 左右双闭 [left, right]
            if not exist_new_seed:
                consecutive_sameness += 1
            else:
                consecutive_sameness = 0

            # 2.3 合并 (收集到区间 + memory中未在本轮中继续拓展的区间)
            old_intervals = [interval for idx, interval in enumerate(memory_intervals) if idx not in found_indices_in_memory]
            merged_intervals = self._merge_intervals(intervals + old_intervals)

            # 3.top-k填充
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index.scatter_(dim=1, index=seed_indices, value=True)
            x[transfer_index] = x0[transfer_index]
            # x.scatter_(dim=1, src=x0, index=seed_indices) 天坑！！！ scatter方法的意思是从src中顺序地选择赋入x的index位置处。index指定的是x中要被赋值的位置而不是在src中要被选择用于赋值的位置。

            # 4. 历史区间更新
            memory_intervals = merged_intervals

            # 研究用：中间状态收集
            steps_used += 1
            # print(f"types: x0-{x0.dtype}, confidence-{confidence.dtype}, transfer_index: {transfer_index.dtype}")
            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])
            history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in merged_intervals])
            # print(f"探索步骤 {exp_step + 1}: 找到区间 {merged_intervals}")

            # 5. 探索阶段提前结束判断
            if consecutive_sameness >= 2:
                has_effective_expansion = False
                for i in range(len(merged_intervals)):
                    curr_start, curr_end = merged_intervals[i]
                    prev_start, prev_end = memory_intervals[i]
                    if 1.0 * (curr_end - curr_start + 1) / (prev_end - prev_start + 1) > 0.05:
                        has_effective_expansion = True
                if not has_effective_expansion: #若无新区间且旧区间增长<=0.05，则提前结束探索
                    break

        # 基于密度的再次扩散
        # print(f"探索阶段找到原始区间: {memory_intervals}")
        desities = [conf_temp[:, start:end+1].nan_to_num(neginf=0).mean().item() for start, end in memory_intervals]
        expected_deltas = [int(desities[i] * (end - start + 1) // 2) for i, (start, end) in enumerate(memory_intervals)]
        density_expansion_intervals = []  # [(new_left, new_right), ...]
        for i, (start, end) in enumerate(memory_intervals):
            delta = expected_deltas[i]
            # 左侧搜索范围
            left_search_min = max(prompt_len, start - delta)
            left = start
            for pos in range(start - 1, left_search_min - 1, -1):
                if x[0, pos] != self.mask_id:
                    break
                left = pos
            # 右侧搜索范围
            right_search_max = min(x.shape[1] - 1, end + delta)
            right = end
            for pos in range(end + 1, right_search_max + 1):
                if x[0, pos] != self.mask_id:
                    break
                right = pos
            density_expansion_intervals.append((left, right))
        memory_intervals = self._merge_intervals(density_expansion_intervals)

        history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in density_expansion_intervals])

        # print(f"边界扩张后找到最终区间: {memory_intervals}")
        return x, memory_intervals, steps_used, outputs, confidences, transfer_idxs, history_intervals

    def acceleration_phase(
            self,
            x: Tensor,
            prompt_index: Tensor,
            intervals: List[tuple[int, int]],
            acceleration_method='factor'
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
        interval_states = [{'coords': (start, end), 'status': 'active'} for start, end in intervals]

        # 状态收集，用于可视化
        outputs = []
        confidences = []
        transfer_idxs = []

        # 只要还存在活跃的区间，就继续
        while any(s['status'] == 'active' for s in interval_states):
            # 动态创建只包含当前要加速的区间的mask
            dynamic_accel_mask = torch.zeros_like(x, dtype=torch.bool)
            active_intervals_found = False
            for state in interval_states:
                if state['status'] == 'active':
                    start, end = state['coords']
                    dynamic_accel_mask[:, start: end + 1] = True
                    active_intervals_found = True
            # 如果所有活跃区间都已填满但状态未更新，则退出
            # if not active_intervals_found or not (dynamic_accel_mask & (x == mask_id)).any():
            #     break
            if not active_intervals_found:
                break

            # 执行一次forward并行更新
            x0, confidence = self._model_forward(x, prompt_index)

            confidence_in_active_zones = torch.where(dynamic_accel_mask, confidence, -np.inf)
            # 更新策略1: 并行解码策略更新
            # TODO: 2.考虑在不同区间内结合各自的密度进行更新，可以基于区间密度平均(背诵vs灵感)，也可以将密度作为置信值的一部分(熟就是熟)，或二者都(数学式)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            if acceleration_method == 'threshold':
                transfer_index = (confidence_in_active_zones > self.acceleration_threshold)
            elif acceleration_method == 'factor':
                # TODO: 3.使用Fast-dLLM中的公式: (n + 1) * (1 - c_{n}) < f 来确定最大的可并行解码n
                # 1. 对>min_threshold的位置按confidence排序; 3. 对这些满足条件的index形成transfer_inedx
                cand_threshold = 0.0 * self.acceleration_threshold + 1.0 * self.acceleration_low_threshold
                cand_mask = (confidence_in_active_zones > cand_threshold)  # (L,)
                # print(f"LB-satisfied #tokens: {cand_mask.sum().item()}")
                cand_idxs = torch.nonzero(cand_mask, as_tuple=False)[:, 1]  # (n,)
                cand_confs = confidence_in_active_zones[cand_mask]  # (n,)
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
            # 更新策略2: 若区间内元素不达解码阈值但仍高于最低阈值(可以直接是探索区间的阈值), 则进行top-k保守(conservative)更新
            n_conf_updated = transfer_index.sum().item()
            n_cons_updated = 0
            if n_conf_updated < self.min_k:
                confidence_for_topk = confidence_in_active_zones.clone()
                # 排除已经为True的位置
                if n_conf_updated > 0:
                    confidence_for_topk[transfer_index] = -np.inf
                _, topk_index = torch.topk(confidence_for_topk, dim=1, k = self.min_k - n_conf_updated)
                # 最低置信度检测
                cons_mask = confidence_for_topk.gather(dim=1, index=topk_index) > self.acceleration_low_threshold
                cons_index = topk_index[cons_mask].unsqueeze(0)  # topk_index[topk_mask]会将结果自动降维
                n_cons_updated = cons_mask.sum().item()
                # 增加填充位置
                transfer_index.scatter_(dim=1, index=cons_index, value=True)
            # 高置信度更新无法进行，保守更新也无法进行。退出加速阶段
            if n_conf_updated + n_cons_updated == 0:
                break
            x[transfer_index] = x0[transfer_index]
            # print(f"n_updated({n_conf_updated + n_cons_updated}) = n_conf_updated({n_conf_updated}) + n_cons_updated({n_cons_updated})")
            # TODO: 3.根据区间边缘的密度，在一定程度上再动态推进区间（惯性）
            # 更新每个活跃区间的状态
            for state in interval_states:
                if state['status'] == 'active':
                    start, end = state['coords']
                    # 检查这个区域是否还有[MASK]
                    if not (x[:, start: end + 1] == self.mask_id).any():
                        state['status'] = 'completed'
                        # print(f"区间 {state['coords']} 加速完成!")

            # dev: 更新历史记录
            steps_used += 1  # 统计步数
            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_idxs.append(transfer_index.detach().cpu().numpy()[0][prompt_len:])

        return x, steps_used, outputs, confidences, transfer_idxs

    def mop_up_phase(
            self,
            x: Tensor,
            prompt_index: Tensor
    ):
        """
            阶段三：收尾阶段
            用传统的top-k方式，填充所有剩余的[MASK]位置。
            speed: 每次decode的[MASK]数量，默认为2
        """

        todo_steps = math.ceil((x == self.mask_id).sum(dim=1).item() / self.mopup_speed)
        todo_steps = min(todo_steps, self.max_mopup_steps)

        outputs = []
        confidences = []
        transfer_idxs = []
        steps_used = 0

        if todo_steps == 0:
            # print("Mopup phase: all tokens decoded, skip mop_up phase.")
            return x, steps_used, outputs, confidences, transfer_idxs

        prompt_len = prompt_index[0].sum().item()
        mask_index = (x == self.mask_id)
        num_transfer_tokens = get_num_transfer_tokens(mask_index, todo_steps)  # (b, todo_steps)

        for step_i in range(todo_steps):
            x0, confidence = self._model_forward(x, prompt_index)
            steps_used += 1

            _, topk_idxs = torch.topk(confidence, k=num_transfer_tokens[0, step_i], dim=1)  # (b, l)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index.scatter_(dim=1, index=topk_idxs, value=True)
            x[transfer_index] = x0[transfer_index]

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
    ):
        """
        实现“多区域并行置信度驱动解码”思路的主函数。
        """
        # 初始化
        start_time = time.perf_counter()

        assert gen_length <= self.model_max_genlength, f"gen_length must <= model_max_genlength({self.model_max_genlength})"
        assert max_steps <= self.model_max_steps, f"steps must <= model_max_steps({self.model_max_steps})"
        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = (x != self.mask_id)

        # initalize positional weights
        position_weights = self._precompute_positional_weights(
            max_steps=max_steps, gen_length=gen_length, max_weight=1, initial_min_weight=0.25,
            device=self.model.device, dtype=torch.float32
        )

        # 主循环 (探索与加速)
        outputs = []
        confidences = []
        transfer_idxs = []
        phase_states = []  # [{'phase':'exploration/acceleration/mopup', 'range': (start, end)}]
        exploration_intervals = []  # [{'inceptive_step': 0, 'history_intervals': [[(start, end), ...], [(start, end), ...], ...]}]
        accumulated_steps = 0

        # print(f"Starting Inference ============================= {x.shape}")

        for EA_idx in range(max_steps - self.max_mopup_steps):
            # 检查是否大部分已经解码完成
            num_masked = (x == self.mask_id).sum().item()
            masked_ratio = 1.0 * num_masked / gen_length
            if masked_ratio < (1 - self.mopup_gate_ratio):
                # print(f"已解码{(1 - masked_ratio) * 100}% (>{mopup_gate_ratio * 100}%), 退出E-A阶段")
                break

            # ① 探索阶段
            x, intervals, exploration_steps, exploration_outputs, exploration_confidences, exploration_transfer_idxs, history_intervals \
                = self.exploration_phase(x, prompt_index, position_weights, EA_idx, accumulated_steps)

            # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Exploration")
            outputs.extend(exploration_outputs)
            confidences.extend(exploration_confidences)
            transfer_idxs.extend(exploration_transfer_idxs)
            phase_states.append(
                {'phase': 'exploration', 'range': (accumulated_steps, accumulated_steps + exploration_steps)})
            exploration_intervals.append({'inceptive_step': accumulated_steps, 'history_intervals': history_intervals})
            # print(f"exploration phase ends, use steps: {exploration_steps}, TPS: {(num_masked - num_masked_exploration) / (exploration_steps)}")
            # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Exploration")

            accumulated_steps += exploration_steps
            # ② 加速阶段
            x, acceleration_steps, outputs_acceleration, confidences_acceleration, transfer_idxs_acceleration \
                = self.acceleration_phase(x, prompt_index, intervals, 'factor')

            # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Acceleration")
            outputs.extend(outputs_acceleration)
            confidences.extend(confidences_acceleration)
            transfer_idxs.extend(transfer_idxs_acceleration)
            phase_states.append(
                {'phase': 'acceleration', 'range': (accumulated_steps, accumulated_steps + acceleration_steps)})
            accumulated_steps += acceleration_steps
            # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Acceleration")
            # TODO: 根据当前自信度(第1和第2答案的logits均值)，决定进入收尾还是直接commit答案

        # ③ 收尾阶段
        # print("\n--- 进入收尾阶段 ---")
        x, mopup_steps, outputs_mopup, confidences_mopup, transfer_idxs_mopup \
            = self.mop_up_phase(x, prompt_index)
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

        total_steps = accumulated_steps + mopup_steps
        end_time = time.perf_counter()
        duration = end_time - start_time
        actual_steps = total_steps

        print(f"The whole decoding ends!")
        print(f"Use steps: {actual_steps}")
        if actual_steps > 0:
            print(f"Reduced steps: {max_steps / actual_steps:.2f}X")
            print(f"Tokens per second: {max_steps / duration:.2f}")
        print(f"Used time: {duration:.2f}s")

        return x, outputs, confidences, transfer_idxs, phase_states, exploration_intervals


def main():
    device = 'cuda:0'
    model_path = "../models/LLaDA-8B-Instruct"
    # model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    prompts = gsm8k_dataset['test']['question'][0:3]

    # --- 使用类进行生成 ---
    max_gen_steps = 256
    sampler = MRSampler.from_path(
        model_path=model_path,
        device='cuda:0',
        cfg_scale=0.,
        temperature=0,
        max_exploration_steps=5,
        N=2,
        M=3,
        exploration_threshold=0.1,
        acceleration_threshold=0.8,
        acceleration_low_threshold=0.5,
        max_mopup_steps=10,
        mopup_gate_ratio=0.85,
    )

    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)
        tokenizer = sampler.tokenizer

        m = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)


        # 2. 调用 generate 方法
        out, outputs, confidences, transfer_idxs, phase_states, exploration_intervals \
            = sampler.generate(input_ids, gen_length=max_gen_steps, max_steps=max_gen_steps)

        ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Prompt_{i}'s answer: {ans}\n")


if __name__ == '__main__':
    main()