import warnings

import numpy as np
import torch
from torch import Tensor

from datasets import load_dataset
from matplotlib import patches, gridspec
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize

import os
import shutil
from dllm.DiCo import DiCo

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

def visualize_overall_steps(
        outputs:list,
        confidences:np.ndarray|list,
        transfer_idxs:np.ndarray|list,
        sink_mask:np.ndarray|list=None,
        prompt:str='', 
        answer:str = '', 
        is_show:bool=False, 
        is_save:bool=True, 
        output_filename:str="", 
        phase_records=None,
        color_map='Blues',
        fallback_steps:list=None
    ):

    DPI = 100
    num_steps = len(confidences)
    seq_len = len(confidences[0])

    fig, ax = plt.subplots(figsize=(seq_len * 0.4, num_steps * 0.3))
    # fontsize = max(8, 18 - seq_len // 10)  # 动态字体大小
    fontsize = 8

    # 使用imshow绘制热力图. rasterized=True启用栅格化渲染，显著提升大图的渲染效率
    im = ax.imshow(confidences, cmap=color_map, interpolation='nearest', vmin=0, vmax=1, aspect='auto', rasterized=True)

    # 在每个小方格中添加Token文本
    for i in range(num_steps):
        for j in range(seq_len):
            # 根据背景色的深浅，决定文字用黑色还是白色，以保证清晰可读
            bg_color_val = confidences[i][j]
            text_color = "w" if bg_color_val > 0.6 else "black"
            # text_color = 'black'
            ax.text(j, i, outputs[i][j],
                    ha="center", va="center", color=text_color, fontsize=fontsize)

            # transfer token框红：检查当前步骤(i)的当前位置(j)的transfer值是否为True
            edgecolor = 'green' if fallback_steps is not None and i in fallback_steps else 'red'
            if transfer_idxs[i][j]:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    linewidth=5,          # 设置边框线宽
                    edgecolor=edgecolor,      # 设置边框颜色为红色/粉色/绿色
                    facecolor='none'      # 设置填充色为无，只保留边框
                )
                ax.add_patch(rect)

            # sink token框为黄色
            if sink_mask is not None and sink_mask[i][j]:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    linewidth=5,          # 设置边框线宽
                    edgecolor='yellow',      # 设置边框颜色为黄色
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
    tick_spacing_x = 8
    tick_spacing_y = 4
    ax.set_xticks(np.arange(0, seq_len, tick_spacing_x))
    ax.set_yticks(np.arange(0, num_steps, tick_spacing_y))
    ax.set_xticklabels(np.arange(0, seq_len, tick_spacing_x)) 
    ax.set_yticklabels(np.arange(0, num_steps, tick_spacing_y)) 

    # 调整布局防止标签重叠
    fig.tight_layout()


    # 若分阶段，则用大框标识区别各阶段 (待for DiCo)
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
        if 'history_intervals_all' in phase_records:
            for interval_data in phase_records['history_intervals_all']:
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
    if is_show:
        # 在 Jupyter 中显示时使用低 DPI 以减小输出大小
        PREVIEW_DPI = 50
        fig.set_dpi(PREVIEW_DPI)
        # print(f"Preview DPI: {PREVIEW_DPI}")
        plt.show()
    if is_save:
        # 保存时使用高 DPI
        # print(f"Save DPI: {DPI}")
        # 如果output_filename不以常见图像格式结尾，则添加.png后缀
        if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf')):
            output_filename += ".pdf"
        plt.savefig(f"{output_filename}", bbox_inches='tight')
    plt.close() #防止在控制台中打印


def visualize_single_step(
        step_idx:int,
        layer_outputs:list,
        layer_confidences:np.ndarray|list,
        sink_mask:np.ndarray|list=None,
        transfer_idxs:np.ndarray|list=None,
        prompt:str='',
        answer:str = '',
        is_show:bool=False, 
        is_save:bool=True, 
        output_filename:str=""
    ):
    """ 
        单步可视化, 带tuned_lens和attn_sink 
        :param step_idx: 当前步骤索引
        :param layer_outputs: (N_layers + 1, seq_len) 每层的输出token列表 (tuned lens/logit lens)
        :param layer_confidences: (N_layers + 1, seq_len) 每层的置信度列表 (tuned lens/logit lens)
        :param sink_mask: (N_layers + 1, seq_len) 每层的sink token mask列表 
        :param transfer_idxs: (N_layers + 1, seq_len) 每层的transfer token列表 
    """
    DPI = 50
    
    n_layers = len(layer_outputs)
    seq_len = len(layer_outputs[0])

    fig, ax = plt.subplots(figsize=(seq_len * 0.5, n_layers * 0.3))
    # fontsize = max(8, 18 - seq_len // 10)  # 动态字体大小
    fontsize = 8

    # 使用imshow绘制热力图
    im = ax.imshow(layer_confidences, cmap='Blues', interpolation='nearest', vmin=0, vmax=1, aspect='auto', rasterized=True)

    # 在每个小方格中添加Token文本
    for i in range(n_layers):
        for j in range(seq_len):
            # 根据背景色的深浅，决定文字用黑色还是白色，以保证清晰可读
            bg_color_val = layer_confidences[i][j]
            text_color = "w" if bg_color_val > 0.6 else "black"
            # text_color = 'black'
            ax.text(j, i, layer_outputs[i][j],
                    ha="center", va="center", color=text_color, fontsize=fontsize)

            # sink token框为黄色
            if sink_mask is not None and i < len(sink_mask) and sink_mask[i][j]:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    linewidth=5,          # 设置边框线宽
                    edgecolor='yellow',      # 设置边框颜色为黄色
                    facecolor='none'      # 设置填充色为无，只保留边框
                )
                ax.add_patch(rect)
            
            # 最后一个layer的transfer token框红
            if i == n_layers - 1 and transfer_idxs[j]:
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
    ax.set_title(f"******Step {step_idx} of the Prompt******: \n"
                 f"{prompt if len(prompt)<=2000 else prompt[:2000]+'...'} \n"
                 f"******Answer******: {answer if len(answer) <=2000 else answer[:2000]}", fontsize=fontsize)

    # 设置刻度，使其显示在每个方格的中心
    tick_spacing_x = 10
    tick_spacing_y = 5
    ax.set_xticks(np.arange(0, seq_len, tick_spacing_x))
    ax.set_yticks(np.arange(0, n_layers, tick_spacing_y))
    ax.set_xticklabels(np.arange(1, seq_len + 1, tick_spacing_x))  # 标签从1开始
    ax.set_yticklabels(np.arange(1, n_layers + 1, tick_spacing_y))  # 标签从1开始

    # 调整布局防止标签重叠
    fig.tight_layout()

    if is_show:
        # 在 Jupyter 中显示时使用低 DPI 以减小输出大小
        PREVIEW_DPI = 50
        fig.set_dpi(PREVIEW_DPI)
        print(f"Preview DPI: {PREVIEW_DPI}")
        plt.show()
    if is_save:
        # 保存时使用高 DPI
        print(f"Save DPI: {DPI}")
        # 如果output_filename不以常见图像格式结尾，则添加.png后缀. 默认存为pdf矢量图.
        if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf')):
            output_filename += ".pdf"
        plt.savefig(f"{output_filename}", bbox_inches='tight')
    plt.close() #防止在控制台中打印

    

# 辅助函数2: 绘制全部的预测结果，并高亮某一步
def plot_decoding_history_on_ax(ax, tokens, transfers, confidences=None, img_cache=None, step_idx=-1, prompt_len=0):
    """
    在一个给定的matplotlib Axes对象(ax)上绘制完整的解码历史热力图。
    并用边框高亮显示当前step所在的行。
    """
    assert confidences is not None or img_cache is not None

    num_steps, gen_len = np.array(tokens).shape

    # 动态调整字体大小，防止在格子很小时文字溢出
    fontsize = max(2, 16 - gen_len // 6)

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
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

    # 高亮框出当前step对应行
    highlight_rect = patches.Rectangle(
        (-0.5, step_idx - 0.5), # 矩形左下角坐标
        gen_len,                          # 矩形宽度
        height=1,                          # 矩形高度
        linewidth=2,
        edgecolor='lime',                 # 使用鲜绿色，易于区分
        facecolor='none'
    )
    ax.add_patch(highlight_rect)

    xticks = np.arange(gen_len)
    xlabels = xticks + prompt_len
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=fontsize)
    display_yticks = np.array([1, step_idx, num_steps])
    ax.set_yticks(display_yticks - 1)
    ax.set_yticklabels([str(tick) for tick in display_yticks], fontsize=fontsize)
    ax.set_title(f"Full Decoding History (Current Step {step_idx + 1})", fontsize=16)
    ax.tick_params(axis='x', labeltop=True, labelbottom=False) # 将X轴刻度置于顶部

# 辅助函数3: 绘制单个Attention Map
def plot_single_attention_map_on_ax(ax, attention_map_data, title, prompt_len, transfers, show_axis=True):
    """
    在一个给定的matplotlib Axes对象(ax)上绘制一个注意力图。
    这是从 visualize_attention_maps 中抽取的单个图绘制逻辑。
    """

    ax.imshow(attention_map_data, cmap='viridis', origin="upper", vmin=0, vmax=0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_title(title)

    seq_len = attention_map_data.shape[0]
    demasked_positions = np.where(transfers)[0] + prompt_len

    # 在 prompt_len 位置绘制红色虚线以分割区域
    # print(f"plot_single_attention_map_on_ax: prompt_len {prompt_len}")
    ax.axvline(x=prompt_len - 0.5, color='red', linestyle='--', linewidth=3)
    ax.axhline(y=prompt_len - 0.5, color='red', linestyle='--', linewidth=3)
    for pos in demasked_positions:
        ax.axhline(y=pos - 0.5, color='#CCC', linestyle='-', linewidth=1)
        ax.axhline(y=pos + 0.5, color='#CCC', linestyle='-', linewidth=1)

    if show_axis:
        # 检查 prompt_len 是否在有效范围内
        assert 0 <= prompt_len <= seq_len
        # 设置刻度，只显示0，最大序列长，和prompt_len
        base_ticks = np.array([0, seq_len, prompt_len]) - 0.5  #-0.5是为了让刻度显示在正中间
        demasked_ticks = demasked_positions - 0.5
        xticks = np.concatenate((base_ticks, demasked_ticks))
        # 格子的起始位置
        base_labels = ['0', str(seq_len), str(prompt_len)]
        demasked_labels = demasked_positions.astype(str)
        tick_labels = np.concatenate((base_labels, demasked_labels))
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])



