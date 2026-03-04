import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

class DLLMAttnVisualizer:
    """
    专门为 dLLM 研究设计的注意力演化看板
    支持维度: [Steps, Layers, Heads, Seq, Seq]
    """
    def __init__(self, attn_weights:np.ndarray, decoded_tokens=None, pred_confidences=None, prompt_len=-1):
        """
        :param attn_weights: 5D Numpy array 或 Tensor (Steps, Layers, Heads, Seq, Seq)
        :param decoded_tokens: 可选, 总的tuned_lens或logit_lens解码token列表 (Steps, Layers, Seq)
        :param pred_confidences: 可选, 总的预测置信度列表 (Steps, Layers, Seq)
        :param prompt_len: 可选, prompt长度, 用于在可视化中区分prompt和生成部分
        """
        # 确保数据是 numpy 格式
        if hasattr(attn_weights, 'cpu'):
            self.attn = attn_weights.detach().cpu().numpy()
        else:
            self.attn = attn_weights
            
        self.decoded_tokens = decoded_tokens
        self.pred_confidences = pred_confidences
        self.prompt_len = prompt_len
        self.n_steps, self.n_layers, self.n_heads, self.seq_len, _ = self.attn.shape
        self._setup_widgets()

    def _setup_widgets(self):
        # 样式设置
        style = {'description_width': 'initial'}
        layout = widgets.Layout(width='300px')

        # 控件定义
        self.slider_step = widgets.IntSlider(
            value=0, min=0, max=self.n_steps-1, 
            description='Diffusion Step:', style=style, layout=layout
        )
        self.select_layers = widgets.SelectMultiple(
            options=list(range(self.n_layers)), value=(0, self.n_layers//2, self.n_layers-1),
            description='Layers (Ctrl选):', style=style, layout=widgets.Layout(width='300px', height='150px')
        )
        self.select_heads = widgets.SelectMultiple(
            options=list(range(self.n_heads)), value=(0, 1),
            description='Heads:', style=style, layout=widgets.Layout(width='300px', height='100px')
        )
        self.check_avg = widgets.Checkbox(value=True, description='Show Layer-Head Avg', style=style)
        self.show_tokens = widgets.Checkbox(
            value=True,
            description='Show Decoded Tokens',
            style=style,
        )
        
        # 输出区域
        self.output_area = widgets.Output()

        # 绑定事件
        for widget in [self.slider_step, self.select_layers, self.select_heads, self.check_avg, self.show_tokens]:
            widget.observe(self._update_dashboard, 'value')

    def _update_dashboard(self, change=None):
        step = self.slider_step.value
        layers = self.select_layers.value
        heads = self.select_heads.value
        show_avg = self.check_avg.value
        show_tokens = self.show_tokens.value

        with self.output_area:
            clear_output(wait=True)
            
            # --- 计算并显示全局总 Avg (所有 Layer, 所有 Head) ---
            # 维度是 (Layers, Heads, Seq, Seq)，在 0, 1 轴上取均值
            global_avg_map = self.attn[step].mean(axis=(0, 1))
            
            fig_global, ax_global = plt.subplots(figsize=(5, 4), dpi=80)
            im_global = ax_global.imshow(global_avg_map, cmap='viridis', interpolation='nearest')
            ax_global.set_title(f"Step {step}: Global Average (All Layers & Heads)", fontsize=12, fontweight='bold')
            plt.colorbar(im_global, ax=ax_global)
            # 在prompt_len位置绘制虚线
            if self.prompt_len > 0:
                ax_global.axvline(x=self.prompt_len - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax_global.axhline(y=self.prompt_len - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_global.axis('off')
            plt.show() # 先把总图打出来

            # -------------------------------------------------------

            if not layers:
                print("请在左侧选择想要查看的 Layer。")
                return

            # --- Layer/Head 细节网格 ---
            n_cols = (1 if show_avg else 0) + len(heads)
            if n_cols == 0:
                print("请至少选择一个 Head，或打开 Layer-Head Avg。")
                return

            rows_per_layer = 1
            n_rows = len(layers) * rows_per_layer

            fig = plt.figure(figsize=(n_cols * 3, n_rows * 3), dpi=80)
            gs = fig.add_gridspec(n_rows, n_cols)

            # axes[i, j] 只对 attn 行有效；token 行单独用 gs span
            axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                             for r in range(n_rows)])

            for i, l_idx in enumerate(layers):
                row_attn = i * rows_per_layer

                # 打印当前layer的attn
                col_off = 0
                if show_avg:
                    ax = axes[row_attn, 0]
                    avg_map = self.attn[step, l_idx].mean(axis=0)
                    ax.imshow(avg_map, cmap='viridis', interpolation='nearest')
                    ax.set_title(f"Layer{l_idx} Head-Avg", fontsize=9)
                    # 在prompt_len位置绘制虚线
                    if self.prompt_len > 0:
                        ax.axvline(x=self.prompt_len - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
                        ax.axhline(y=self.prompt_len - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
                    ax.axis('off')
                    col_off = 1
                
                for j, h_idx in enumerate(heads):
                    ax = axes[row_attn, j + col_off]
                    ax.imshow(self.attn[step, l_idx, h_idx], cmap='viridis', interpolation='nearest')
                    ax.set_title(f"Layer{l_idx} H{h_idx}", fontsize=9)
                    # 在prompt_len位置绘制虚线
                    if self.prompt_len > 0:
                        ax.axvline(x=self.prompt_len - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
                        ax.axhline(y=self.prompt_len - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
                    ax.axis('off')

                # 在attn下面打印当前layer的tuned lens decoded_tokens
                # 用热力图打印，前景是token文字，背景是置信度，viridis配色

                
            plt.tight_layout()
            plt.show()
            plt.close('all') # 释放所有内存

    def show(self):
        """显示看板"""
        controls = widgets.VBox([
            widgets.HTML("<b>dLLM Attention Analyzer</b>"),
            self.slider_step, 
            self.select_layers, 
            self.select_heads, 
            self.check_avg,
            self.show_tokens,
        ])
        # 简单的边框装饰
        controls.layout.margin = '0 20px 0 0'
        ui = widgets.HBox([controls, self.output_area])
        display(ui)
        self._update_dashboard() # 初始触发一次