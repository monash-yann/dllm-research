import csv
import re
import torch
from datasets import load_dataset
from dllm.DLLMExpr import DLLMExpr, BaselineConfig

    
# gsm8k prompt
gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
device = 'cuda:1'
# use llada
model_path = "/home/xiangzhong_ayl/dllm/works/dllm-research/models/LLaDA-8B-Instruct"
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
    max_weight=1.0,
    initial_min_weight=0.05,
    remasking="low_confidence",
    decoding_method="fixed",
    factor=1,
    k=1,
    confidence_threshold=0.9,
    **token_info
)
sampler = DLLMExpr.from_path(
    model_path=model_path,
    device=device,
    config=config,
    torch_dtype=torch.bfloat16
)
# sampler.set_length_strategy(DAEDAL())   # dynamic length

tokenizer = sampler.tokenizer



def extract_gsm8k_answer(text):
    """
    完全复刻 lm-eval 的 GSM8K 提取逻辑，并新增优先提取 \boxed{} 的逻辑：
    0. 优先尝试提取末尾 \boxed{xxx} 中的内容
    1. 失败则尝试 'strict-match' (#### 后的数字)
    2. 再次失败则使用 'flexible-extract' (提取文中最后一个看起来像数字的片段)
    """
    if not text:
        return "[invalid]"

    # --- 策略 0: Boxed Match (新增) ---
    # 匹配所有的 \boxed{...} 并取最后一个，确保即便后面有干扰数字也能优先命中
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        # 取最后一个 boxed 里的内容，并去掉逗号以便后续数值比较
        return boxed_matches[-1].replace(",", "")

    # --- 策略 1: Strict Match (对应配置中的 strict-match) ---
    strict_pattern = r"####\s?(-?[0-9\.\,]+)"
    strict_match = re.search(strict_pattern, text)
    if strict_match:
        # 提取捕获组内容，并去掉逗号以便后续数值比较
        return strict_match.group(1).replace(",", "")

    # --- 策略 2: Flexible Extract (对应配置中的 flexible-extract) ---
    # 逻辑: 寻找所有可能的数字格式，取最后一个匹配项 (group_select: -1)
    flexible_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    # findall 返回 list of tuples, e.g., [('$100', ''), ('', '5')]
    matches = re.findall(flexible_pattern, text)
    
    if matches:
        # group_select: -1 -> 取列表中的最后一个匹配 (通常是最终答案)
        last_match = matches[-1]
        
        # 处理 tuple: 正则有 A|B 两个组，取其中非空的那个
        extracted_val = next((m for m in last_match if m), None)
        
        if extracted_val:
            # 清理格式：去掉尾部空格句点，去掉非数值字符($, ,)
            return extracted_val.strip(" .").replace(",", "").replace("$", "")

    return "[invalid]"


def generate_under_various_lengths_gsm8k(qid=2, start=64, end=512, inc=16):
    """
    Docstring for generate_under_various_lengths_gsm8k. 
    生成多个不同生成长度的结果以探寻生成结构与长度的关系, 将结果保存为csv文件.
    结果文件含三列: "gen_length", "gen_output", "extracted_answer"
    
    :param qid: 题目ID, 从gsm8k test中获取
    :param start: 最小生成长度
    :param end: 最大生成长度
    :param inc: 生成长度的增量
    """

    block_length = 32
    prompt_text = gsm8k_dataset['test']['question'][qid]
    prompt_text += "\nWrap the final answer in a \\boxed{}."    #确保答案输出
    expected_resp = gsm8k_dataset['test']['answer'][qid]
    m = [{"role": "user", "content": prompt_text}]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    expected_length = tokenizer(expected_resp, return_tensors="pt").input_ids.shape[1]
    expected_ans = extract_gsm8k_answer(expected_resp)

    output_file = f"visualization/huashan2/data/various_lengths_expr/gsm8k_qid_{qid}_{start}to{end}_inc{inc}.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([f"gen_length(exp={expected_length})", "gen_output", f"extracted_answer(exp={expected_ans})"])
        
        # 开始循环
        for gen_length in range(start, end + 1, inc):
            OUT = sampler.generate(prompt=input_ids, gen_length=gen_length, max_steps=gen_length, block_length=block_length, records=[])
            resp = tokenizer.batch_decode(OUT.out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            extracted_ans = extract_gsm8k_answer(resp)
            
            # 4写入 CSV 并打印日志 同时Replace newline in ans to keep csv clean
            # resp = resp.replace("\n", "\\n") 
            writer.writerow([gen_length, resp, extracted_ans])  
            # 立即刷入硬盘。防止程序在第 500 步崩了，前 499 步白跑
            f.flush() 
            
            # print(f"qid: {qid} | len: {gen_length} | extracted: {extracted_ans}")


if __name__ == "__main__":
    # do multiple samples
    for qid in range(0, 5):  
        print(f"--- Generating for GSM8K Question ID: {qid} ---")
        generate_under_various_lengths_gsm8k(qid=qid, start=32, end=512, inc=2)

    # one sample, 长度精细控制
    # qid = 2
    # print(f"--- Generating for GSM8K Question ID: {qid} ---")
    # generate_under_various_lengths_gsm8k(qid=qid, start=385, end=512, inc=1)






# --- 测试用例 (验证逻辑) ---
# examples = [
#     # Case 1: 标准格式 (Strict Match 生效)
#     "The answer is #### -12,345", 
#     # Case 2: 无 ####，取最后一个数字 (Flexible Match 生效)
#     "I have 10 apples, sold 5. So I have -5.", 
#     # Case 3: 包含货币符号的复杂格式
#     "The cost is $1,000.50.",
#     # Case 4: 只有单个数字
#     "-42" 
#     # Case 5: 包含 \boxed{} 的格式 (Boxed Match 生效)
#     "Therefore, Kylar needs to pay \\(\\boxed{64}\\) dollars for the 16 glasses.",
# ]

# for ex in examples:
#     print(f"Input: {ex:<40} | Extracted: {extract_gsm8k_answer(ex)}")