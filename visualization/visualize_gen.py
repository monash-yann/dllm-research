import numpy as np
import torch

from visualizer import get_local
from datasets import load_dataset
import os
import shutil
from matplotlib import patches, gridspec
import matplotlib.pyplot as plt

from sampler.MRSampler import MRSampler, MRSamplerConfig, GenerateOutput
from sampler.PureLLaDASampler import PureLLaDASamplerConfig, PureLLaDASampler
from sampler.utils import decode_outputs
from utils import visualize_overall_steps, plot_decoding_history_on_ax, plot_single_attention_map_on_ax


# 绘制控制函数:
def run_gen_until(
    sampler: MRSampler,
    prompts: list,
    gen_length: int,
    max_steps: int,
    block_length: int,
    output_dir: str,
    vis_overall=True, vis_attn_map=True, console_show=False, file_save=True,
    device: str = 'cuda',
    **kwargs
):
    """
        根据prompts生成回答并可视化
    """
    output_dir = os.path.abspath(output_dir)    #转绝对路径
    os.makedirs(output_dir, exist_ok=True)
    overall_output_dir = os.path.join(output_dir, "overall")
    if file_save and vis_overall:
        if os.path.exists(overall_output_dir):
            shutil.rmtree(overall_output_dir)
        os.makedirs(overall_output_dir, exist_ok=True)


    model = sampler.model
    tokenizer = sampler.tokenizer

    print(f"visualize to path: {os.path.abspath(output_dir)}")
    key = 'LLaDABlock._manually_scaled_dot_product_attention'   #attention_map key
    steps_for_all = 0
    genlength_for_all = 0

    for i, prompt in enumerate(prompts, 1):

        get_local.cache[key] = []
        print(f"generating answer {i} with max_step-{max_steps}, gen_length={gen_length} to {output_dir}")
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        OUT: GenerateOutput = sampler.generate(input_ids, gen_length=gen_length, max_steps=max_steps, block_length=block_length)
        out = OUT.out
        outputs = OUT.outputs
        phase_states = OUT.phase_states
        exploration_intervals = OUT.exploration_intervals
        confidences = OUT.confidences
        transfer_idxs = OUT.transfer_idxs
        metrics = OUT.metrics

        print(f"prompt_{i} decoded over, metrics={metrics}")
        actual_steps = metrics.use_steps
        actual_genlength = len(outputs[0]) if len(outputs)>0 else 0
        steps_for_all += actual_steps
        genlength_for_all += actual_genlength

        outputs_decoded = decode_outputs(outputs, tokenizer)
        # 可视化阶段信息准备
        phase_records = None
        if phase_states is not None and exploration_intervals is not None:
            phase_records = {
                'phase_states': phase_states,
                'exploration_intervals': exploration_intervals,
            }
        n_prompt_tokens = input_ids.shape[1]

        # steps输出结果整体 可视化
        answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        if vis_overall:

            visualize_overall_steps(overall_output_dir, (outputs_decoded, confidences, transfer_idxs),
                i, prompt, answer, is_show=console_show, is_save=file_save, phase_records=phase_records)
            print(f"overall saved to {overall_output_dir}")

        # 逐 Step Attention Maps 可视化
        cache = get_local.cache
        if vis_attn_map:
            existed_imgs_step = 0
            attn_map_output_dir = ''
            if file_save:
                attn_map_output_dir = os.path.join(output_dir, f"Q{i}_details")
                if os.path.exists(attn_map_output_dir):
                    shutil.rmtree(attn_map_output_dir)
                os.makedirs(attn_map_output_dir, exist_ok=True)

                # existed_imgs_step = len(os.listdir(attn_map_output_dir))
                # if existed_imgs_step == actual_steps:
                #     print(f"prompt_{i} already done, go to next step")
                #     continue
                # elif existed_imgs_step > 0:
                #     print(f"prompt_{i} partially done, continue generating from step{existed_imgs_step + 1}")
                # else:
                #     print(f"prompt_{i} fresh, starting generating its attention_maps")


            value_list = cache[key]
            # print(f"---{len(value_list)}---")
            assert len(value_list) % actual_steps == 0
            n_layers = len(value_list) // actual_steps

            shape_per_layer = value_list[0].shape # (batch_size, heads, seq_len, seq_len)
            steps_attention_maps = np.array(value_list).reshape(
                (actual_steps, n_layers) + shape_per_layer
            ).astype(np.float16) # (steps, N_layers, batch_size, heads, seq_len, seq_len)
            # print("steps_attention_maps shape:", steps_attention_maps.shape)

            # --- 动态计算布局参数 ---
            # FIG_WIDTH = max(32, gen_length * 0.7)
            FIG_WIDTH = 32
            P1_HEIGHT = actual_steps * (FIG_WIDTH / actual_genlength) # decoding_history区域高度
            P2_HEIGHT = FIG_WIDTH * 3 / 4 # attention_maps区域高度
            P3_HEIGHT = 0 # prompt&answer区域高度
            FIG_HEIGHT = P1_HEIGHT + P2_HEIGHT

            # 绘制attention maps on each step
            for step_idx in range(existed_imgs_step, actual_steps):
                fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
                # 由上至下共3个区域
                gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[P1_HEIGHT, P2_HEIGHT, P3_HEIGHT], hspace=0.2)

                # --- 1. 绘制上方的decoding_history大图 ---
                ax_history = fig.add_subplot(gs_main[0, :]) # 占据第一行的所有列
                plot_decoding_history_on_ax(
                    ax=ax_history,
                    tokens=outputs_decoded,
                    confidences=confidences,
                    transfers=transfer_idxs,
                    step_idx=step_idx,
                    prompt_len=n_prompt_tokens
                )

                # --- 2. 绘制下方的attention_maps ---
                gs_p2 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs_main[1, 0], hspace=0.2)
                # 中间画一张大的总平均attention_map。在N_layers和heads两个维度上取平均
                ax_avg_all = fig.add_subplot(gs_p2[:, :])
                attn_data_all_avg = steps_attention_maps[step_idx, :, 0].mean(axis=(0, 1)) # shape: (seq, seq)
                plot_single_attention_map_on_ax(
                    ax=ax_avg_all,
                    attention_map_data=attn_data_all_avg,
                    title=f"All {n_layers} Layers Avg",
                    prompt_len=n_prompt_tokens,
                    transfers=transfer_idxs[step_idx]
                )

                plt.suptitle(f"******Answer******: {answer if len(answer) <= 2000 else answer[:2000]}", fontsize=16)

                if file_save:
                    save_path = os.path.join(attn_map_output_dir, f"step_{step_idx + 1}.png")
                    plt.savefig(save_path)
                    # print(f"attention_map saved to {save_path}")
                if console_show:
                    plt.show()

                plt.close(fig) # 必须关闭，否则会在内存中累积

            if file_save:
                print(f"prompt{i}'s attention_maps have been saved to {attn_map_output_dir}")

        # 清空已处理过的attention_map，为下一个prompt做准备
        if key in cache:
            cache[key] = []

    print(f"{len(prompts)} promtps over. avg steps reduced: {steps_for_all / len(prompts):.2f}X")

def visualize_MR():
    print(f"visualizing MR Sampler, current path: {os.path.abspath(__file__)}")
    device = 'cuda:2'

    # 提示词替换，与4-shot的行为保持一致
    few_shot_filename = "../prompts/gsm8k_shot.txt"
    gsm8k_prompts = []
    with open(few_shot_filename, "r", encoding="utf-8") as f:
        for line in f:
            # python会把.txt中的字符当作原始字符串，此处转为普通字符串
            corrected_line = line.replace('\\n', '\n')
            gsm8k_prompts.append(corrected_line)

    # gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    # gsm8k_prompts = gsm8k_dataset['test']['question'][:10]

    # get_local.activate()  # 在引入模型之前，激活装饰器
    model_path = "../models/LLaDA-8B-Instruct"
    config = MRSamplerConfig(
        cfg_scale=0.0,
        temperature=0.0,
        max_exploration_steps=10,
        exploration_N=3,
        exploration_M=2,
        exploration_threshold=0.25,
        acceleration_parallel_method='fixed',
        acceleration_threshold=0.9,
        acceleration_low_threshold=0.6,
        acceleration_factor=1,
        max_mopup_steps=10,
        mopup_gate_ratio=0.85,
        mopup_speed=2,
        positional_weights_type='none',
        max_weight=1.0,
        initial_min_weight=0.1,
    )

    sampler = MRSampler.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )

    # exploration_thresholds = [0.05, 0.15, 0.3, 0.5]
    # exploration_thresholds = [0.2, 0.25]  -> No Positionals Weights下, 0.25表现最好
    # exploration_thresholds = [0.25, 0.1, 0.15, 0.35]
    # exploration_thresholds = [0.15, 0.35]
    exploration_thresholds = [0.25]

    sampler.positional_weights_type = 'none'

    gen_length = 128
    block_length = 64
    output_dir = f"imgs/dico+V1/gsm8k_L${gen_length}_B${block_length}/N{sampler.exploration_N}E{sampler.max_exploration_steps}_APM{sampler.acceleration_parallel_method}_PWT{sampler.positional_weights_type}_imw${sampler.initial_min_weight}"
    run_gen_until(
        sampler=sampler,
        prompts=gsm8k_prompts[:5],
        max_steps=gen_length,
        gen_length=gen_length,
        block_length=block_length,
        output_dir=output_dir,
        device=device,
        console_show=False, file_save=True, vis_overall=True, vis_attn_map=False
    )

def visualize_pure_llada():
    print(f"visualizing pure llada, current path: {os.path.abspath(__file__)}")
    device = 'cuda:0'

    # 提示词替换，与4-shot的行为保持一致
    # few_shot_filename = "../prompts/gsm8k_shot.txt"
    # gsm8k_prompts = []
    # with open(few_shot_filename, "r", encoding="utf-8") as f:
    #     for line in f:
    #         # python会把.txt中的字符当作原始字符串，此处转为普通字符串
    #         corrected_line = line.replace('\\n', '\n')
    #         gsm8k_prompts.append(corrected_line)
    # gsm8k_prompts = gsm8k_prompts[:1]

    # 普通0-shot提示词
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    gsm8k_prompts = gsm8k_dataset['test']['question'][0:1]


    get_local.activate()  # 在引入模型之前，激活装饰器
    model_path = "../models/LLaDA-8B-Instruct"
    config = PureLLaDASamplerConfig(
        cfg_scale=0.0,
        temperature=0.0,
        positional_weights_type='none',
        max_weight=1.0,
        initial_min_weight=0.0,
        remasking="low_confidence",
        decoding_method="topk",
        k=2
    )

    gen_length = 64
    block_length = 64
    sampler = PureLLaDASampler.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )

    output_dir = f"imgs/pure-MTD${sampler.decoding_method}_${sampler.k if sampler.k!=-1 else ''}/gsm8k_L${gen_length}_B${block_length}/"
    run_gen_until(
        sampler=sampler,
        prompts=gsm8k_prompts,
        max_steps=gen_length,
        gen_length=gen_length,
        block_length=block_length,
        output_dir=output_dir,
        device=device,
        console_show=False, file_save=True, vis_overall=True, vis_attn_map=True,
    )

if __name__ == "__main__":
    # visualize_MR()
    visualize_pure_llada()

