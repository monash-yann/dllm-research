import numpy as np
import torch

from tool import TunedLens
from transformers import AutoModel
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
    model = sampler.model

    # np读取rawdata/hidden_states_gsm8k_pmt0.npy文件
    hidden_states_all = np.load('visualization/huashan2/rawdata/hidden_states_gsm8k_pmt0.npy')
    hidden_states_step0_ts = torch.tensor(hidden_states_all[0])  # 取第一步的hidden states，shape (n_layers, seq_len, hidden_size)

    tuned_lens = TunedLens.from_model_and_pretrained_lens(
        model=model,
        lens_path='tool/tuned_lens/checkpoints/model.safetensors'
    )
    tuned_lens.eval()

    decoded_outputs = []
    for i in range(hidden_states_step0_ts.shape[0]):
        if i >= hidden_states_step0_ts.shape[0] - 1: continue
        try:
            # 注意：Tuned Lens forward 需要 [Batch, Seq, Dim]
            h = hidden_states_step0_ts[i].unsqueeze(0).to(dtype=model.dtype)
            mid_logits = tuned_lens.forward(h, i) # [1, Seq, Vocab]
            
            # 从MODEL_PATH加载tokenizer，并对每个mid_logits进行解码
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            decoded_output = tokenizer.decode(torch.argmax(mid_logits, dim=-1)[0], skip_special_tokens=True)
            decoded_outputs.append(decoded_output)
            print(f"Layer {i} decoded output: {decoded_output}")
        except IndexError: continue
    # print(f"Decoded outputs from Tuned Lens at step 0:\n{decoded_outputs}")

     

if __name__ == '__main__':
    main()