import sys
from pathlib import Path
import os
import gc
import numpy as np
import torch
from datasets import load_dataset
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Since it's a script, no GUI needed
import matplotlib.pyplot as plt

# ---- Path Setup ----
repo_root = Path.cwd()
while repo_root != repo_root.parent and not (repo_root / "dllm").exists():
    repo_root = repo_root.parent
if (repo_root / "dllm").exists():
    sys.path.insert(0, str(repo_root))

from dllm.DLLMExpr import BaselineConfig, DLLMExpr
from dllm.utils import decode_outputs
from visualization.utils.utils import visualize_overall_steps, visualize_single_step
from tool.tuned_lens.tuned_lens import TunedLens
from dllm.DLLM import GenerateOutput
from tool.attn_sink import sink_metric_epsilon


if __name__ == "__main__":
    
    # ==========================================
    # 1. Model & Tokenizer Initialization
    # ==========================================
    device = 'cuda:0'
    model_path = f"{repo_root}/models/LLaDA-8B-Instruct"
    token_info = {
        'mask_id': 126336,
        'bos_id': 126080,
        'pad_id': 126081,
        'eos_id': 126081,
        'eot_id': 126348
    }
    config = BaselineConfig(
        cfg_scale=0.0,
        temperature=0.0,
        positional_weights_type='none',
        remasking="low_confidence",
        decoding_method="fixed",
        factor=1,
        k=1,
        confidence_threshold=0.9,
        **token_info
    )

    print("Loading Model...")
    sampler = DLLMExpr.from_path(
        model_path=model_path,
        device=device,
        config=config,
        torch_dtype=torch.bfloat16
    )
    tokenizer = sampler.tokenizer
    sampler.model.to(device)

    print("Loading Tuned Lens...")
    tuned_lens = TunedLens.from_model_and_pretrained_lens(
        model=sampler.model,
        lens_path=f'{repo_root}/tool/tuned_lens/checkpoints/model.safetensors',
        device=device
    )
    tuned_lens.eval()

    # ==========================================
    # 3. Data Loading & Generation
    # ==========================================
    block_length = 32
    gen_length = 256
    qid = 2

    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    prompt_text = gsm8k_dataset['test']['question'][qid]

    m = [{"role": "user", "content": prompt_text}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    print("Starting Generation...")
    OUT: GenerateOutput = sampler.generate(
        prompt=input_ids, gen_length=gen_length, max_steps=gen_length, block_length=block_length, 
        records=['metrics', 'state_trace'], tuned_lens=tuned_lens
    )

    metrics = OUT.metrics
    outputs = OUT.state_trace['outputs_all'] # (T, Seq)
    confidences = OUT.state_trace['confidences_all']
    transfer_idxs = OUT.state_trace['transfer_idxs_all']
    attentions = OUT.state_trace['attentions_all']
    hidden_states = OUT.state_trace['hidden_states_all']
    answer = tokenizer.batch_decode(OUT.out[:, prompt_len:], skip_special_tokens=True)[0]
    outputs_decoded = decode_outputs(outputs, tokenizer)

    fallback_steps = OUT.state_trace['fallback_steps']
    print(f"Fallback steps count: {len(fallback_steps)}")
    print(f"metrics={metrics}")

    # ==========================================
    # Attn Sink Computation
    # ==========================================
    print("Calculating attention sinks...")
    sink_metrics = sink_metric_epsilon(attentions, epsilon=3)
    sink_head_avg = sink_metrics['sink_mask_head_avg'].numpy()  # (T, L, S)
    sink_layer_head_avg = sink_metrics['sink_mask_layer_head_avg'].numpy()  # (T, S)

    # ==========================================
    # Overall Visualization
    # ==========================================
    print("Plotting overall visualization...")
    output_dirname = f"{repo_root}/visualization/huashan2/imgs/decodings/gsm8k_qid_{qid}_GL{gen_length}BL{block_length}_TD"
    os.makedirs(output_dirname, exist_ok=True)

    output_filename = f"{output_dirname}/gsm8k_qid_{qid}_GL{gen_length}BL{block_length}.pdf"
    visualize_overall_steps(
        outputs_decoded, confidences, transfer_idxs, sink_mask=sink_layer_head_avg,
        prompt=prompt, answer=answer, 
        is_show=False, is_save=True, output_filename=output_filename,
        color_map='Blues', fallback_steps=fallback_steps
    )
    print(f"Overall visualization saved to {output_filename}")

    # ==========================================
    # Tuned Lens Visualization
    # ==========================================
    print("Analyzing layer-wise distributions via Tuned Lens...")
    lens_type = 'tuned_lens'
    hidden_states_all = OUT.state_trace['hidden_states_all']   
    T, n_layers, seq_len, hs = hidden_states_all.shape
    n_layers -= 1  

    decoded_tokens_all = []
    pred_confidences_all = []
    eos_probs_steps_layers = []   

    for step_idx in range(T):
        hidden_states_step_ts = torch.tensor(hidden_states_all[step_idx])  # (n_layers, seq_len, hidden_size)
        decoded_tokens_step = []
        pred_confidences_step = []
        eos_probs_step = []

        with torch.no_grad():
            for layer_idx in range(0, n_layers):
                h = hidden_states_step_ts[layer_idx].unsqueeze(0).type(sampler.model.dtype).to(device)  # [1, Seq, Hidden]
                if lens_type == 'tuned_lens' and layer_idx < n_layers - 1:
                    mid_logits = tuned_lens.forward(h, layer_idx)  # [1, Seq, Vocab]
                else:
                    mid_logits = tuned_lens.unembed(h)  # logit lens
                
                probs = torch.softmax(mid_logits, dim=-1)   # (1, Seq, Vocab)
                max_probs, max_ids = torch.max(probs, dim=-1)  # (1, Seq)
                
                layer_decoded = []
                for pos in range(seq_len):
                    token_text = tokenizer.decode(max_ids[0, pos])
                    layer_decoded.append(token_text)
                decoded_tokens_step.append(layer_decoded)
                pred_confidences_step.append(max_probs[0].cpu().type(torch.float32).numpy())
                # eos_probs = probs[:, :, eos_id]  # (1, Seq)
                # eos_probs_step.append(eos_probs[0].cpu().type(torch.float32).numpy())

        x0_decoded = []
        for pos in range(seq_len):
            token_text = tokenizer.decode(outputs[step_idx, pos])
            x0_decoded.append(token_text)
        decoded_tokens_step.append(x0_decoded)
        pred_confidences_step.append(confidences[step_idx])
        # eos_probs_step.append(eos_probs_steps[step_idx])    
            
        decoded_tokens_all.append(decoded_tokens_step)
        pred_confidences_all.append(pred_confidences_step)
        # eos_probs_steps_layers.append(eos_probs_step)   
        
    pred_confidences_all = np.array(pred_confidences_all)  # (T, n_layers, seq_len)
    # eos_probs_steps_layers = np.array(eos_probs_steps_layers)  # (T, n_layers, seq_len)

    # Padding for initial embedding sink
    embedding_layer_pad = np.expand_dims(np.zeros_like(sink_head_avg[:, 0]), axis=1)  # (T, 1, S)
    sink_mask_layers = np.concatenate([embedding_layer_pad, sink_head_avg], axis=1)

    step_idxs_vis = fallback_steps
    for step_idx in step_idxs_vis:
        out_filename = f"{output_dirname}/gsm8k_qid_{qid}_GL{gen_length}BL{block_length}_step_{step_idx}.pdf"
        visualize_single_step(
            step_idx=step_idx, 
            layer_outputs=decoded_tokens_all[step_idx], layer_confidences=pred_confidences_all[step_idx],
            sink_mask=sink_mask_layers[step_idx], transfer_idxs=transfer_idxs[step_idx],
            prompt=prompt, answer=answer, 
            is_show=False, is_save=True, output_filename=out_filename
        )

    # ==========================================
    # 6. Unembedding & EOS Probability Matrix
    # ==========================================
    # print("Unembedding hidden states to compute EOS probabilities...")
    # temp_device = device 
    # hidden_states_steps = torch.tensor(hidden_states[:, -1, :, :]).type(torch.bfloat16)   # (T, S, H)
    # T, S, H = hidden_states_steps.shape
    # probs_steps = []
    # sz = 8 
    # unembedding_matrix = sampler.model.get_output_embeddings().to(temp_device) 
    # for t in range(0, T, sz):
    #     hidden_states_batch = hidden_states_steps[t: t+sz].to(temp_device)
    #     unembeded_batch = unembedding_matrix(hidden_states_batch).softmax(dim=-1)
    #     probs_steps.append(unembeded_batch.type(torch.float32).to('cpu'))  # (sz, S, V)
    # probs_steps = torch.cat(probs_steps, dim=0)  # (T, S, V)

    # del unembedding_matrix
    # del hidden_states_steps
    # gc.collect()
    # torch.cuda.empty_cache()

    # eos_id = token_info['eos_id']
    # eos_probs_steps = probs_steps[:, :, eos_id]  # (T, S)

    # plt.figure(figsize=(S * 0.5, T * 0.5))
    # plt.imshow(eos_probs_steps, aspect='auto', cmap='Blues', vmin=0, vmax=1, rasterized=False)
    # plt.colorbar(label='EOS Probability')

    # plt.xticks(np.arange(0, S, 32))
    # plt.yticks(np.arange(0, T, 32))
    # plt.xlabel('positions')
    # plt.ylabel('steps')
    # ax = plt.gca()
    # ax.set_xticklabels(np.arange(0, S, 32))
    # ax.set_yticklabels(np.arange(0, T, 32))
    # plt.title(f'EOS Probability Heatmap Gen_length={gen_length} Block_length={block_length}')

    # if prompt_len > 0:
    #     plt.axvline(x=prompt_len - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=1)

    # plt.tight_layout()
    # plt.savefig(f'{output_dirname}/eos_probs.pdf')
    # plt.close()

    # ==========================================
    # 8. Render EOS Heatmaps per Step
    # ==========================================
    # print("Rendering EOS heatmaps and single-step visualizations...")
    # eos_id = token_info['eos_id']
    # T, L, S = eos_probs_steps_layers.shape
    # step_idxs = [0, 1, 2]
    # for idx, step_idx in enumerate(step_idxs):
    #     plt.figure(figsize=(S * 0.5, L * 0.5))
    #     plt.imshow(eos_probs_steps_layers[step_idx], aspect='auto', cmap='Blues', vmin=0, vmax=1, rasterized=False)
    #     plt.colorbar(label='EOS Probability')

    #     plt.xticks(np.arange(0, S, 32))
    #     plt.yticks(np.arange(0, L, 4))
    #     plt.xlabel('positions')
    #     plt.ylabel('layers')
    #     plt.title(f'EOS Probability Heatmap Gen_length={gen_length} Block_length={block_length}, Step={step_idx}')

    #     if prompt_len > 0:
    #         plt.axvline(x=prompt_len - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=1)

    #     plt.tight_layout()
    #     plt.savefig(f'{output_dirname}/tunedlens_eos_probs_step{step_idx}.pdf')
    #     plt.close()


    print("All done!")
