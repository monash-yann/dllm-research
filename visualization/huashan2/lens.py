import numpy as np
import torch
import matplotlib.pyplot as plt

from tool import TunedLens
from transformers import AutoTokenizer

from dllm import DLLMExpr
from dllm.DLLMBaseline import BaselineConfig

def main():
    MODEL_PATH="/home/xiangzhong_ayl/dllm/models/LLaDA-8B-Instruct"
    token_info = {
        'mask_id': 126336,
        'bos_id': 126080,
        'pad_id': 126081,
        'eos_id': 126081,
        'eot_id': 126348
    }
    config = BaselineConfig(
        **token_info,
        remasking="low_confidence",
        positional_weights_type='none',
        decoding_method="topk",
        factor=1,
        k=1,
        confidence_threshold=0.9,
    )
    sampler = DLLMExpr.from_path(model_path=MODEL_PATH, config=config)
    tokenizer = sampler.tokenizer
    model = sampler.model

    tuned_lens = TunedLens.from_model_and_pretrained_lens(
        model=sampler.model,
        lens_path='tool/tuned_lens/checkpoints/model.safetensors'
    )
    tuned_lens.eval()
    
    # gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    # prompts = gsm8k_dataset['test']['question'][0:1]

    step_idxs = [0]
    hidden_states_all = np.load('visualization/huashan2/rawdata/hidden_states_gsm8k_pmt0.npy')
    for step_idx in step_idxs:
        # np读取rawdata/hidden_states_gsm8k_pmt0.npy文件
        hidden_states_step_ts = torch.tensor(hidden_states_all[step_idx])  # shape: (n_layers, seq_len, hidden_size)

        n_layers, seq_len, _ = hidden_states_step_ts.shape
        token_grid = [["" for _ in range(seq_len)] for _ in range(n_layers)]
        conf_grid = np.zeros((n_layers, seq_len), dtype=np.float32)

        with torch.no_grad():
            for layer_idx in range(n_layers - 1):
                try:
                    # 注意：Tuned Lens forward 需要 [Batch, Seq, Dim]
                    h = hidden_states_step_ts[layer_idx].unsqueeze(0).to(dtype=sampler.model.dtype)
                    mid_logits = tuned_lens.forward(h, layer_idx)  # [1, Seq, Vocab]

                    probs = torch.softmax(mid_logits, dim=-1)
                    max_probs, max_ids = torch.max(probs, dim=-1)  # [1, Seq]

                    for pos in range(seq_len):
                        token_text = tokenizer.decode([int(max_ids[0, pos])], skip_special_tokens=True)
                        if token_text == "":
                            token_text = "<empty>"
                        token_grid[layer_idx][pos] = token_text
                        conf_grid[layer_idx, pos] = float(max_probs[0, pos])
                except IndexError:
                    continue

        fig_w = max(8.0, seq_len * 0.35)
        fig_h = max(6.0, n_layers * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(conf_grid, origin='lower', aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)

        for layer_idx in range(n_layers):
            for pos in range(seq_len):
                text_color = 'white' if conf_grid[layer_idx, pos] < 0.6 else 'black'
                ax.text(pos, layer_idx, token_grid[layer_idx][pos], ha='center', va='center', color=text_color, fontsize=7)

        ax.set_xlabel('Position (seq_len)')
        ax.set_ylabel('Layer (n_layers)')
        ax.set_title(f'Tuned Lens Decoded Tokens (step {step_idx})')
        fig.colorbar(im, ax=ax, label='Softmax Confidence')

        output_path = f'visualization/huashan2/imgs/tuned_lens_tokens_step{step_idx}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=250)
        plt.show()

     

if __name__ == '__main__':
    main()