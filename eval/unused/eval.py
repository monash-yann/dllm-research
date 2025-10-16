import argparse
import json
import shutil

import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from peft import PeftModel

from sampler.MRSampler import MRSampler
from eval.unused.tasks.gsm8k import GSM8KDataset, parse_gsm_answer

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    # "math": MATH500Dataset,
    # "countdown": CTDDataset,
    # "svamp": SVAMPDataset,
}

PARSE_MAP = {
    "gsm8k": parse_gsm_answer,
    # "math": parse_math_answer,
    # "countdown": parse_ctd_answer,
    # "svamp": parse_svamp_answer,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(
    sampler: MRSampler,
    dataloader: DataLoader,
    gen_length: int=256,
    steps: int=256,
    block_length: int=256,
    enable_vote=False,
    parse_answer_func=None,
):
    model = sampler.model
    tokenizer = sampler.tokenizer
    sampler.N = 1
    sampler.max_exploration_steps = 20

    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out = sampler.generate(input_ids, gen_length=gen_length, max_steps=steps)[0]

        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=True)
        # print(f"ans: {generated_texts[0]}")

        if enable_vote:
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    # "vote_answer": vote_answers[j],
                    "final_answer": parse_answer_func(generated_texts[j]), 
                    "ground_truth": gt_answers[j],
                }
                for j in range(len(gt_answers))
            ]
        else:
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                }
                for j in range(len(gt_answers))
            ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")
            if enable_vote:
                print("-" * 50)
                # print(f"Vote answer: {vote_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../models/LLaDA-8B-Instruct/")
    parser.add_argument("--model_name", type=str, default="LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "math", "countdown", "svamp", "human_eval_plus"], default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation.")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="cfg_scale for generation.")

    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")

    parser.add_argument(
        "--enable_vote",
        action="store_true",
        help="Whether to enable vote functionality."
    )

    parser.add_argument(
        "--vote_method",
        type=str,
        choices=["fixed", "linear", "exp"],
        default=None,
        help="Voting method to use: 'fixed', 'linear', or 'exp'."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha parameter used for 'exp' voting method."
    )

    args = parser.parse_args()

    num_evals = {"gsm8k": 2, "math": -1, "svamp": -1, "countdown": -1}

    sampler = MRSampler.from_path(
        model_path=args.model_path,
        device=f'cuda:{local_rank}',
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        max_exploration_steps=10,
        N=2,
        M=3,
        exploration_threshold=0.15,
        acceleration_threshold=0.8,
        acceleration_low_threshold=0.6,
        max_mopup_steps=10,
        mopup_gate_ratio=0.9,
    )

    # output_dir = os.path.abspath(f"{args.output_dir}_N2E10")
    output_dir = os.path.abspath(f"{args.output_dir}_test")

    print(f"Sampling Args: {args}")
    print(f"Result will be saved to path: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    if args.checkpoint_path:
        model = PeftModel.from_pretrained(sampler.model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    dataset = DATASET_MAP[args.dataset](
        sampler.tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill for all models
    )

    
    if args.enable_vote:
        parse_func = PARSE_MAP.get(args.dataset, None)
    else:
        parse_func = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = args.model_name

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    filename = f"{output_dir}/rank_{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")


    metrics = evaluate(
        sampler,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        enable_vote=args.enable_vote,
        parse_answer_func=parse_func,
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )

    cleanup_ddp()
