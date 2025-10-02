import warnings

from visualizer import get_local

from datasets import load_dataset
from sampler.Sampler import MRSampler
from sampler.utils import decode_outputs
from utils import *
import os
import shutil
from matplotlib import patches, gridspec
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 绘制控制函数:
def run_gen_until(
    sampler: MRSampler,
    prompts: list,
    gen_length: int, max_steps: int,
    output_dir: str,
    vis_overall=True, vis_attn_map=True, console_show=False, file_save=True,
    **kwargs
):
    """
        根据prompts生成回答并可视化
    """
    output_dir = os.path.abspath(output_dir)    #转绝对路径
    os.makedirs(output_dir, exist_ok=True)
    overall_output_dir = os.path.join(output_dir, "overall")
    if os.path.exists(overall_output_dir):
        shutil.rmtree(overall_output_dir)

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

        out, outputs, confidences, transfer_idxs, phase_states, exploration_intervals \
            = sampler.generate(input_ids, gen_length=gen_length, max_steps=max_steps)

        print(f"prompt_{i} decoded over for {output_dir}/prompt_{i}")
        actual_steps = len(outputs)
        actual_genlength = len(outputs[0]) if len(outputs)>0 else 0
        steps_for_all += actual_steps
        genlength_for_all += actual_genlength
        # 批量demask一下outputs
        outputs_decoded = decode_outputs(outputs, tokenizer)
        # 可视化阶段信息准备
        phase_records = None
        if phase_states is not None and exploration_intervals is not None:
            phase_records = {
                'phase_states': phase_states,
                'exploration_intervals': exploration_intervals,
            }
        n_prompt_tokens = len(input_ids)

        # steps输出结果整体 可视化
        answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        if vis_overall:
            if file_save:
                os.makedirs(overall_output_dir, exist_ok=True)
            visualize_overall_steps(overall_output_dir, (outputs_decoded, confidences, transfer_idxs),
                i, prompt, answer, is_show=console_show, is_save=file_save, phase_records=phase_records)
            print(f"overall saved to {overall_output_dir}")

        # 逐 Step Attention Maps 可视化
        cache = get_local.cache
        if vis_attn_map:
            value_list = cache[key]
            assert len(value_list) % actual_steps == 0
            n_layers = len(value_list) // actual_steps
            shape_per_layer = value_list[0].shape # (batch_size, heads, seq_len, seq_len)
            # 重新塑形，将steps维度提取出来
            steps_attention_maps = np.array(value_list).reshape(
                (actual_steps, n_layers) + shape_per_layer
            ).astype(np.float16) # (steps, N_layers, batch_size, heads, seq_len, seq_len)
            # --- 动态计算布局参数 ---
            # 固定绘图的宽度
            # FIG_WIDTH = max(32, gen_length * 0.7)
            FIG_WIDTH = 16
            P1_HEIGHT = actual_steps * (FIG_WIDTH / actual_genlength) # decoding_history区域高度
            P2_HEIGHT = FIG_WIDTH * 3 / 4 # attention_maps区域高度
            P3_HEIGHT = 1 # prompt&answer区域高度
            FIG_HEIGHT = P1_HEIGHT + P2_HEIGHT

            # 为当前prompt的每个step生成一张包含预测和所有层注意力的大图
            prompt_output_dir = os.path.join(output_dir, f"prompt_{i}_details")
            # if os.path.exists(prompt_output_dir):
            #     shutil.rmtree(prompt_output_dir)
            os.makedirs(prompt_output_dir, exist_ok=True)
            existed_imgs_step = len(os.listdir(prompt_output_dir))
            if existed_imgs_step == actual_steps:
                print(f"prompt_{i} already done, go to next step")
                continue
            elif existed_imgs_step > 0:
                print(f"prompt_{i} partially done, continue generating from step{existed_imgs_step + 1}")

            for step_idx in range(existed_imgs_step, actual_steps):
                # 创建一个足够大的画布
                fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

                # 使用GridSpec创建 3x1 的布局，分别对应P1(decoding_history)和P2(attention_maps)
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
                gs_p2 = gridspec.GridSpecFromSubplotSpec(6, 8, subplot_spec=gs_main[1, 0], hspace=0.2)
                # 中间画一张大的总平均attention_map。在N_layers和heads两个维度上取平均
                ax_avg_all = fig.add_subplot(gs_p2[0:6, 1:7])
                attn_data_all_avg = steps_attention_maps[step_idx, :, 0].mean(axis=(0, 1)) # shape: (seq, seq)

                plot_single_attention_map_on_ax(
                    ax=ax_avg_all,
                    attention_map_data=attn_data_all_avg,
                    title=f"All {n_layers} Layers Avg",
                    prompt_len=n_prompt_tokens,
                    transfers=transfer_idxs[step_idx]
                )

                # --- 3. 在最底部的区域专门放文字信息 ---
                gs_p3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2, 0], wspace=0.1, hspace=0.1)
                # prompt区域
                ax_prompt = fig.add_subplot(gs_p3[0, 0]) # 占据第三行
                ax_prompt.axis('off')
                ax_prompt.text(
                    0.5,  # x坐标 (0=最左, 1=最右)
                    0.5,   # y坐标 (0=最下, 1=最上)
                    f"Prompt: {prompt if len(prompt)<=2000 else prompt[:2000]+'...'}\n ",
                    transform=ax_prompt.transAxes, # 使用相对于ax自身的坐标系，非常方便
                    fontsize=20,
                    horizontalalignment='center', # 新增：水平对齐方式设为居中
                    verticalalignment='center',   # 修改：垂直对齐方式设为居中
                )
                # answer区域
                ax_answer = fig.add_subplot(gs_p3[0, 1]) # 占据第三行
                ax_answer.axis('off')
                ax_answer.text(
                    0.5,  # x坐标 (0=最左, 1=最右)
                    0.5,   # y坐标 (0=最下, 1=最上)
                    f"Answer: {answer if len(answer)<=2000 else answer[:2000]+'...'}\n ",
                    transform=ax_answer.transAxes, # 使用相对于ax自身的坐标系，非常方便
                    fontsize=20,
                    horizontalalignment='center', # 新增：水平对齐方式设为居中
                    verticalalignment='center',   # 修改：垂直对齐方式设为居中
                )

                # 设置总标题
                fig.suptitle(f"Detailed Analysis for Step {step_idx + 1}/{actual_steps}\n", fontsize=24)
                # 保存图像，为每个prompt创建一个子文件夹
                if file_save:
                    save_path = os.path.join(prompt_output_dir, f"step_{step_idx + 1}.png")
                    plt.savefig(save_path)
                    print(f"attention_maps saved to {prompt_output_dir}")
                if console_show:
                    plt.show()

                plt.close(fig) # 必须关闭，否则会在内存中累积

        # 清空已处理过的attention_map，为下一个prompt做准备
        if key in cache:
          cache[key] = []
    print(f"{len(prompts)} promtps over. avg steps reduced: {steps_for_all / m():.2f}X")


if __name__ == "__main__":
    print(f"connected to server, current path: {os.path.abspath(__file__)}")
    device = 'cuda:0'

    # get_local.activate()  # 在引入模型之前，激活装饰器
    # 提示词替换，与4-shot的行为保持一致
    few_shot_filename = "../prompts/gsm8k_shot.txt"
    with open(few_shot_filename, "r", encoding="utf-8") as f:
        gsm8k_shot_prompts= f.readlines()

    model_path = "../models/LLaDA-8B-Instruct"
    model_params = {
        'cfg_scale':  0.0,
        'temperature': 0.0,
        # Exploration phase config
        'max_exploration_steps': 5,
        'N': 2,
        'M': 3,
        'exploration_threshold': 0.15,
        # Acceleration phase config
        'acceleration_threshold': 0.8,
        'acceleration_low_threshold': 0.6,
        'acceleration_factor': 1.0,
        'min_k': 2,
        # Mop-up phase config
        'mopup_gate_ratio': 0.9,
        'max_mopup_steps': 10,
        'mopup_speed': 2,
    }
    # 用户传入的kwargs覆盖默认参数
    sampler = MRSampler.from_path(model_path, device=device, torch_dtype=torch.bfloat16, **model_params)

    # max_exploration_steps_list = [3, 10, 20]
    # for max_exploration_steps in max_exploration_steps_list:
    #     sampler.max_exploration_steps = max_exploration_steps
    #     # 测跑
    #     run_gen_until(
    #         sampler=sampler,
    #         prompts=gsm8k_shot_prompts[:5],
    #         max_steps=256,
    #         gen_length=256,
    #         output_dir=f"./imgs/Test/gsm8k_s256/N2E{max_exploration_steps}_min0.25",
    #         console_show=False, file_save=True, vis_overall=True, vis_attn_map=False
    #     )

    sampler.max_exploration_steps = 10
    exploration_N_list = [1, 5]
    for exploration_N in exploration_N_list:
        sampler.N = exploration_N
        run_gen_until(
            sampler=sampler,
            prompts=gsm8k_shot_prompts[:5],
            max_steps=256,
            gen_length=256,
            output_dir=f"./imgs/Test/gsm8k_s256/N{exploration_N}E20_min0.25",
            console_show=False, file_save=True, vis_overall=True, vis_attn_map=False
        )


