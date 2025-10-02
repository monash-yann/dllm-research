import warnings

import numpy as np
import torch
from visualizer import get_local

from datasets import load_dataset
from matplotlib import patches, gridspec
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize

import os
import shutil
from sampler.Sampler import MRSampler

def sanitize_for_matplotlib(text):
    """更全面地转义 Matplotlib mathtext 的特殊字符"""
    replacements = {
        "\\": r"\\",
        "$": r"\$",
        "_": r"\_",
        "^": r"\^",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def visualize_overall_steps(folder_path:str, OUT:tuple[list, list, list], index:int=0,
        prompt:str='', answer:str = '', is_show:bool=False, is_save:bool=True, phase_records=None):

    BASE_DPI = 90
    MIN_DPI = 70
    BASE_SEQLEN_FOR_DPI = 256
    MAX_SEQLEN_FOR_DPI = 1024
    BM_RATIO = (BASE_DPI - MIN_DPI) / (MAX_SEQLEN_FOR_DPI - BASE_SEQLEN_FOR_DPI)
    outputs, confidences, transfer_idxs = OUT
    # print(f"outputs.shape: {len(outputs)}, {len(outputs[0])}")
    # print(f"confidences.shape: {len(confidences)}, {len(confidences[0])}")
    # print(f"transfer_idxs.shape: {len(transfer_idxs)}, {len(transfer_idxs[0])}")
    num_steps = len(confidences)
    seq_len = len(confidences[0])

    fig, ax = plt.subplots(figsize=(seq_len*1, num_steps*1))
    fontsize = max(8, 18 - seq_len // 10)  # 动态字体大小

    # 使用imshow绘制热力图
    im = ax.imshow(confidences, cmap='Blues', interpolation='nearest', vmin=0, vmax=1, aspect='auto')

    # 在每个小方格中添加Token文本
    for i in range(num_steps):
        for j in range(seq_len):
            # 根据背景色的深浅，决定文字用黑色还是白色，以保证清晰可读
            bg_color_val = confidences[i][j]
            text_color = "w" if bg_color_val > 0.6 else "black"
            # text_color = 'black'
            ax.text(j, i, outputs[i][j],
                    ha="center", va="center", color=text_color, fontsize=fontsize)

            # 关键判断：检查当前步骤(i)的当前位置(j)的transfer值是否为True
            if transfer_idxs[i][j]:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    linewidth=5,          # 设置边框线宽
                    edgecolor='red',      # 设置边框颜色为红色
                    facecolor='none'      # 设置填充色为无，只保留边框
                )
                ax.add_patch(rect)

    # 添加右侧颜色条 (Colorbar)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Confidence Score', rotation=-90, va="bottom")

    # 防止latex公式格式引起的渲染报错
    prompt = sanitize_for_matplotlib(prompt)
    answer = sanitize_for_matplotlib(answer)
    # 设置坐标轴和标签
    ax.set_xlabel("Token Position", fontsize=fontsize)
    ax.set_ylabel("Sampling Step", fontsize=fontsize)
    ax.set_title(f"******Prompt******: \n"
                 f"{prompt if len(prompt)<=2000 else prompt[:2000]+'...'} \n"
                 f"******Answer******: {answer if len(answer) <=2000 else answer[:2000]}", fontsize=fontsize)

    # 设置刻度，使其显示在每个方格的中心
    tick_spacing_x = 10
    tick_spacing_y = 5
    ax.set_xticks(np.arange(0, seq_len, tick_spacing_x))
    ax.set_yticks(np.arange(0, num_steps, tick_spacing_y))
    ax.set_xticklabels(np.arange(1, seq_len + 1, tick_spacing_x))  # 标签从1开始
    ax.set_yticklabels(np.arange(1, num_steps + 1, tick_spacing_y))  # 标签从1开始

    # 调整布局防止标签重叠
    fig.tight_layout()

    # 若分阶段，则用大框标识区别各阶段
    if phase_records is not None:
        # 绘制遍历所有阶段框
        if 'phase_states' in phase_records:
            phase_colors = {
                'exploration': 'lightslategray',  # 蓝灰色
                'acceleration': 'orange',  # 橙色
                'mopup': 'plum'  # 肉色 (贝壳色比较接近)
            }
            for state in phase_records['phase_states']:
                phase_name = state['phase']
                start_step, end_step = state.get('range', (None, None))

                assert start_step is not None and end_step is not None and phase_name in phase_colors
                if end_step - start_step <= 0:
                    continue
                # print(state);
                rect_y = start_step - 0.5
                rect_height = end_step - start_step
                rect = patches.Rectangle(
                    xy=(-0.5, rect_y),  # 左下角坐标 (x=-0.5使其覆盖整行)
                    width=seq_len,  # 宽度为整个图的宽度
                    height=rect_height,  # 高度为阶段所占的步数
                    linewidth=8,  # 边框线宽
                    edgecolor=phase_colors[phase_name],  # 边框颜色
                    facecolor='none',  # 填充颜色
                )
                ax.add_patch(rect)
                # 在每个框的左侧添加文字标签
                ax.text(
                    -0.7,  # x坐标，放在图的左边
                    rect_y + rect_height / 2,  # y坐标，放在框的垂直中心
                    phase_name.capitalize(),  # 标签文字 (首字母大写)
                    rotation=90,  # 旋转90度
                    verticalalignment='center',  # 垂直居中
                    horizontalalignment='right',  # 水平右对齐
                    fontsize=fontsize * 0.8,
                    color=phase_colors[phase_name],
                    fontweight='bold'
                )
        # 绘制探索阶段中每个step确定的区间
        if 'exploration_intervals' in phase_records:
            for interval_data in phase_records['exploration_intervals']:
                inceptive_step = interval_data['inceptive_step']
                # 遍历每行的区间
                for idx, step_intervals in enumerate(interval_data['history_intervals']):
                    current_step = inceptive_step + idx
                    # print(f"于步骤 {current_step}: 渲染区间 {step_intervals}")
                    # 遍历当前步骤的所有区间
                    for start_pos, end_pos in step_intervals:
                        rect = patches.Rectangle(
                            xy=(start_pos - 0.5, current_step - 0.5),  # 矩形左下角坐标
                            width=end_pos - start_pos + 1,  # 区间宽度
                            height=1,  # 高度为1个步骤
                            linewidth=3,  # 边框线宽
                            edgecolor='lightslategray',  # 蓝灰色边框
                            facecolor='none'  # 无填充色
                        )
                        ax.add_patch(rect)
    # 保存图像
    if (is_save):
        dpi = BASE_DPI - BM_RATIO * (seq_len - BASE_SEQLEN_FOR_DPI)
        print(f"dpi :{dpi}")
        plt.savefig(f"{folder_path}/example{index}.png", dpi=dpi)
    if (is_show):
        plt.show()

    plt.close() #防止在控制台中打印

def visualize_attention_maps(attention_map, heads_to_plot: list = None, prompt_len=0, is_show=False, is_save=False, folder_path='', index=-1):
    """
    接收从 get_local.cache 中捕获的注意力图缓存，并将其可视化。

    参数:
    - attention_map (np arr): shape: (batch_size, heads, seq_len, seq_len)
    - heads_to_plot (list, optional): 一个包含要绘制的头索引的列表。
                                      如果为 None，则默认绘制前8个头 [0, 1, ..., 7]。
    """
    # 如果未指定要绘制哪些头，则默认选择前8个
    if heads_to_plot is None:
        heads_to_plot = list(range(8))

    # --- 准备绘图数据 ---
    # 数据形状应为 (1, num_heads, seq_len, seq_len)
    # 我们去掉 batch 维度
    if attention_map.shape[0] != 1:
        warnings.warn(f"输入的 attention map 的 batch size 不为1（当前为 {attention_map.shape[0]}），将只可视化第一个 batch 的数据。")
    attention_heads_data = attention_map[0] # 获取 batch 0 的数据

    num_heads_available = attention_heads_data.shape[0]
    if max(heads_to_plot) >= num_heads_available:
        print(f"错误：请求绘制的头索引 (最大为 {max(heads_to_plot)}) 超出了可用的头数量 ({num_heads_available})。")
        return

    # --- 使用 Matplotlib 绘图 ---
    # 计算网格布局，确保能容纳所有要绘制的头
    num_plots = len(heads_to_plot)
    # 尽可能使用4列，然后计算需要的行数
    ncols = 4
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # 如果只有一个子图，axes 不是一个数组，需要手动转成数组方便索引
    if num_plots == 1:
        axes = np.array([axes])

    # 将多余的子图隐藏
    for i in range(num_plots, nrows * ncols):
        axes.flatten()[i].axis('off')

    for i, head_idx in enumerate(heads_to_plot):
        ax = axes.flatten()[i]

        # 从所有头中获取当前要绘制的那个头的数据
        head_data = attention_heads_data[head_idx]

        plot_single_attention_map_on_ax(ax, head_data, f"head {head_idx}", prompt_len=prompt_len)

    plt.title(f"attention_maps in {num_plots} heads...")
    fig.tight_layout()

    if is_show:
        plt.show()
    if is_save:
        plt.savefig(f"{folder_path}/example{index}.png")
        # print("图像已保存为 attention_map_visualization.png")
    plt.close()

# 辅助函数1: 绘制单步的预测结果
def plot_step_result_on_ax(ax, tokens, confidence, transfers):
    """
    在一个给定的matplotlib Axes对象(ax)上绘制单步的预测结果条。
    这是从 visualize_overall_steps 中抽取的单行绘制逻辑。
    """
    seq_len = len(tokens)

    # imshow需要2D数组，所以我们将1D的confidence变成(1, seq_len)
    confidences_reshaped = np.array(confidence).reshape(1, -1)

    im = ax.imshow(confidences_reshaped, cmap='Blues', interpolation='nearest', vmin=0, vmax=1, aspect='equal', extent=(-0.5, seq_len - 0.5, -0.5, 0.5))

    for j in range(seq_len):
        bg_color_val = confidence[j]
        text_color = "w" if bg_color_val > 0.6 else "black"
        ax.text(j, 0, tokens[j], ha="center", va="center", color=text_color, fontsize=12)
        # 画demask红框
        if transfers[j]:
            rect = patches.Rectangle(
                (j - 0.5, -0.5), 1, 1,
                linewidth=3, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

    ax.set_xticks(np.arange(seq_len))
    ax.set_xticklabels(np.arange(1, seq_len + 1))

    # 自适应白边
    # ax.set_xlim(-0.5, gen_length - 0.5)
    # ax.set_xticks(np.arange(gen_length))
    # ax.set_xticklabels(np.arange(1, gen_length + 1), rotation=90, fontsize=8) # 当画布很宽时，标签旋转防止重叠

    ax.set_yticks([0])
    ax.set_yticklabels(["Pred"])
    ax.set_xlabel("Token Position")

# 辅助函数2: 绘制全部的预测结果，并高亮某一步
def plot_decoding_history_on_ax(ax, tokens, transfers, confidences=None, img_cache=None, step_idx=-1, prompt_len=0):
    """
    在一个给定的matplotlib Axes对象(ax)上绘制完整的解码历史热力图。
    并用边框高亮显示当前step所在的行。
    """
    assert confidences is not None or img_cache is not None

    num_steps, gen_len = np.array(tokens).shape

    # 动态调整字体大小，防止在格子很小时文字溢出
    fontsize = max(4, 16 - gen_len // 6)

    # imshow会自动调整格子大小以适应ax的固定宽度; aspect='equal' 确保每个格子是正方形。
    if img_cache is not None:
        ax.imshow(img_cache, interpolation='nearest', aspect='auto')
    else:
        ax.imshow(confidences, cmap='Blues', interpolation='nearest', vmin=0, vmax=1, aspect='equal')

    # 绘制热力图中的(文字和--del)红框
    for i in range(num_steps):
        for j in range(gen_len):
            bg_color_val = confidences[i][j]
            text_color = "w" if bg_color_val > 0.6 else "black"
            ax.text(j, i, tokens[i][j], ha="center", va="center", color=text_color, fontsize=fontsize)

            if transfers[i][j]:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

    # 新增功能：用一个醒目的边框框出当前step的行
    highlight_rect = patches.Rectangle(
        (-0.5, step_idx - 0.5), # 矩形左下角坐标
        gen_len,                          # 矩形宽度
        1,                                # 矩形高度
        linewidth=4,
        edgecolor='lime',                 # 使用鲜绿色，易于区分
        facecolor='none'
    )
    ax.add_patch(highlight_rect)

    xticks = np.arange(gen_len)
    xlabels = xticks + prompt_len
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=fontsize)
    ax.set_yticks(np.arange(num_steps))
    ax.set_yticklabels(np.arange(1, num_steps + 1), fontsize=fontsize)
    ax.set_title(f"Full Decoding History (Current Step {step_idx})", fontsize=16)
    ax.tick_params(axis='x', labeltop=True, labelbottom=False) # 将X轴刻度置于顶部

# 辅助函数3: 绘制单个Attention Map
def plot_single_attention_map_on_ax(ax, attention_map_data, title, prompt_len, transfers, show_axis=True):
    """
    在一个给定的matplotlib Axes对象(ax)上绘制一个注意力图。
    这是从 visualize_attention_maps 中抽取的单个图绘制逻辑。
    """

    ax.imshow(attention_map_data, cmap='viridis', origin="upper")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_title(title)

    seq_len = attention_map_data.shape[0]
    demasked_positions = np.where(transfers)[0] + prompt_len

    # 在 prompt_len 位置绘制红色虚线以分割区域
    ax.axvline(x=prompt_len - 0.5, color='red', linestyle='--', linewidth=3)
    ax.axhline(y=prompt_len - 0.5, color='red', linestyle='--', linewidth=3)
    for pos in demasked_positions:
        ax.axhline(y=pos - 0.5, color='#CCC', linestyle='-', linewidth=2)
        ax.axhline(y=pos + 0.5, color='#CCC', linestyle='-', linewidth=2)

    if show_axis:
        # 检查 prompt_len 是否在有效范围内
        assert 0 <= prompt_len <= seq_len
        # 设置刻度，只显示0，最大序列长，和prompt_len
        base_ticks = np.array([0, seq_len, prompt_len]) - 0.5  #-0.5是为了让刻度显示在
        demasked_ticks = demasked_positions - 0.5
        xticks = np.concatenate((base_ticks, demasked_ticks))
        # 格子的起始位置
        base_labels = ['0', str(seq_len), str(prompt_len)]
        demasked_labels = demasked_positions.astype(str)
        demasked_set = set(demasked_labels)
        tick_labels = np.concatenate((base_labels, demasked_labels))
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        # 将对应 prompt_len 的刻度标签颜色设置为红色
        # get_xticklabels() 需要在设置xticks后调用才能获取最新的列表
        # xticklabels = ax.get_xticklabels()
        # yticklabels = ax.get_yticklabels()
        # for label in yticklabels:
        #     text = label.get_text()
        #     if text in demasked_set:
        #         label.set_color('green')
        #     elif text == str(prompt_len):
        #         label.set_color('red')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

# 绘制控制函数:
# def run_gen_until(steps, gen_length, block_length, prompts, output_dir,
#       vis_overall=True, vis_attn_map=True, console_show=False, file_save=True):
#     """
#         根据prompts生成回答并可视化
#     """
#     output_dir = os.path.abspath(output_dir)  # 转绝对路径
#     os.makedirs(output_dir, exist_ok=True)
#     overall_output_dir = os.path.join(output_dir, "overall")
#     if os.path.exists(overall_output_dir):
#         shutil.rmtree(overall_output_dir)
#
#     key = 'LLaDABlock._manually_scaled_dot_product_attention'
#     print(f"inferencing and visualizing to path: {os.path.abspath(output_dir)}")
#
#     for i, prompt in enumerate(prompts, 1):
#
#         get_local.cache[key] = []
#         print(f"generating answer {i} with step-{steps}, gen_length={gen_length} to {output_dir}")
#         m = [{"role": "user", "content": prompt}, ]
#         prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
#         input_ids = tokenizer(prompt)['input_ids']
#         input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
#
#         out, outputs_decoded, confidences, transfer_idxs = generate(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
#
#         # steps输出结果整体 可视化
#         answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
#         if vis_overall:
#             if file_save:
#                 os.makedirs(overall_output_dir, exist_ok=True)
#             visualize_overall_steps(overall_output_dir, (outputs_decoded, confidences, transfer_idxs),
#                 i, prompt, answer, is_show=console_show, is_save=file_save)
#             print(f"overall saved to {overall_output_dir}")
#
#         # 逐Step Attention Maps 可视化
#         n_prompt_tokens = len(input_ids)
#         cache = get_local.cache
#         if vis_attn_map:
#             value_list = cache[key]
#             assert len(value_list) % steps == 0
#             n_layers = len(value_list) // steps
#             shape_per_layer = value_list[0].shape # (batch_size, heads, seq_len, seq_len)
#
#             # 重新塑形，将steps维度提取出来
#             steps_attention_maps = np.array(value_list).reshape(
#                 (steps, n_layers) + shape_per_layer
#             ).astype(np.float16) # (steps, N_layers, batch_size, heads, seq_len, seq_len)
#
#             # --- 动态计算布局参数 ---
#             # 固定绘图的宽度
#             # FIG_WIDTH = max(32, gen_length * 0.7)
#             FIG_WIDTH = 64
#             P1_HEIGHT = steps * (FIG_WIDTH / gen_length) # decoding_history区域高度
#             P2_HEIGHT = FIG_WIDTH # attention_maps区域高度
#             P3_HEIGHT = 1 # prompt&answer区域高度
#             FIG_HEIGHT = P1_HEIGHT + P2_HEIGHT
#
#             # 为当前prompt的每个step生成一张包含预测和所有层注意力的大图
#             prompt_output_dir = os.path.join(output_dir, f"prompt_{i}_details")
#             # if os.path.exists(prompt_output_dir):
#             #     shutil.rmtree(prompt_output_dir)
#             os.makedirs(prompt_output_dir, exist_ok=True)
#             existed_imgs_step = len(os.listdir(prompt_output_dir))
#             if existed_imgs_step == steps:
#                 print(f"prompt_{i} already done, go to next step")
#                 continue
#             elif existed_imgs_step > 0:
#                 print(f"prompt_{i} partially done, continue generating from step{existed_imgs_step + 1}")
#             # 优化手段，预绘制并缓存总的cmap热力图。事实证明没啥用
#             # norm = Normalize(vmin=0, vmax=1)
#             # cmap = plt.colormaps.get_cmap('Blues')
#             # decoding_history_image = cmap(norm(confidences))
#
#             for step_idx in range(existed_imgs_step, steps):
#                 # 创建一个足够大的画布
#                 fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
#
#                 # 使用GridSpec创建 2x1 的布局，分别对应P1(decoding_history)和P2(attention_maps)
#                 gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[P1_HEIGHT, P2_HEIGHT, P3_HEIGHT], hspace=0.2)
#
#                 # --- 1. 绘制上方的decoding_history大图 ---
#                 ax_history = fig.add_subplot(gs_main[0, :]) # 占据第一行的所有列
#                 plot_decoding_history_on_ax(
#                     ax=ax_history,
#                     tokens=outputs_decoded,
#                     confidences=confidences,
#                     transfers=transfer_idxs,
#                     step_idx=step_idx,
#                     prompt_len=n_prompt_tokens
#                 )
#
#                 # --- 2. 绘制下方的attention_maps ---
#                 gs_p2 = gridspec.GridSpecFromSubplotSpec(6, 8, subplot_spec=gs_main[1, 0], hspace=0.2)
#                 # 左侧画一张大的总平均attention_map。在N_layers和heads两个维度上取平均
#                 ax_avg_all = fig.add_subplot(gs_p2[0:6, 1:7])
#                 attn_data_all_avg = steps_attention_maps[step_idx, :, 0].mean(axis=(0, 1)) # shape: (seq, seq)
#
#                 plot_single_attention_map_on_ax(
#                     ax=ax_avg_all,
#                     attention_map_data=attn_data_all_avg,
#                     title=f"All {n_layers} Layers Avg",
#                     prompt_len=n_prompt_tokens,
#                     transfers=transfer_idxs[step_idx]
#                 )
#
#                 # 在右侧绘制多张中间layers的heads平均attention_map
#                 # layers_to_plot = np.linspace(0, n_layers - 1, 12, dtype=int)
#                 # for i, layer_idx in enumerate(layers_to_plot):
#                 #     # 在右侧小子ax上绘制当前layer的头平均attention_map
#                 #     ax_nested = fig.add_subplot(gs_p2[i // 6, 2 + i % 6])
#                 #     attn_data_avg = steps_attention_maps[step_idx, layer_idx, 0].mean(axis=0)
#                 #
#                 #     plot_single_attention_map_on_ax(
#                 #         ax=ax_nested,
#                 #         attention_map_data=attn_data_avg,
#                 #         title=f"Layer {layer_idx+1}", # 标题可以简化
#                 #         prompt_len=n_prompt_tokens,
#                 #         transfers=transfer_idxs[step_idx]
#                 #     )
#
#                 # --- 3. 在最底部的区域专门放文字信息 ---
#                 gs_p3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2, 0], wspace=0.1, hspace=0.1)
#                 # prompt区域
#                 ax_prompt = fig.add_subplot(gs_p3[0, 0]) # 占据第三行
#                 ax_prompt.axis('off')
#                 ax_prompt.text(
#                     0.5,  # x坐标 (0=最左, 1=最右)
#                     0.5,   # y坐标 (0=最下, 1=最上)
#                     f"Prompt: {prompt if len(prompt)<=2000 else prompt[:2000]+'...'}\n ",
#                     transform=ax_prompt.transAxes, # 使用相对于ax自身的坐标系，非常方便
#                     fontsize=20,
#                     horizontalalignment='center', # 新增：水平对齐方式设为居中
#                     verticalalignment='center',   # 修改：垂直对齐方式设为居中
#                 )
#                 # answer区域
#                 ax_answer = fig.add_subplot(gs_p3[0, 1]) # 占据第三行
#                 ax_answer.axis('off')
#                 ax_answer.text(
#                     0.5,  # x坐标 (0=最左, 1=最右)
#                     0.5,   # y坐标 (0=最下, 1=最上)
#                     f"Answer: {answer if len(answer)<=2000 else answer[:2000]+'...'}\n ",
#                     transform=ax_answer.transAxes, # 使用相对于ax自身的坐标系，非常方便
#                     fontsize=20,
#                     horizontalalignment='center', # 新增：水平对齐方式设为居中
#                     verticalalignment='center',   # 修改：垂直对齐方式设为居中
#                 )
#
#                 # 设置总标题
#                 fig.suptitle(f"Detailed Analysis for Step {step_idx + 1}/{steps}\n", fontsize=24)
#
#                 # 保存图像，为每个prompt创建一个子文件夹
#                 if file_save:
#                     save_path = os.path.join(prompt_output_dir, f"step_{step_idx + 1}.png")
#                     plt.savefig(save_path)
#
#                 if console_show:
#                     plt.show()
#
#                 plt.close(fig) # 必须关闭，否则会在内存中累积
#             if file_save:
#                 print(f"attention_maps saved to {prompt_output_dir}")
#         # 清空已处理过的attention_map，为下一个prompt做准备
#         if key in cache:
#           cache[key] = []


