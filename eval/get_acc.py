from glob import glob
import json
import re
from eval.tasks.gsm8k import parse_gsm_answer
from eval.utils.parsers import is_equiv

def parse_answer(dataset, text):
    if "gsm" in dataset:
        return parse_gsm_answer(text)
    # elif "svamp" in dataset:
    #     return parse_svamp_answer(text)
    # elif "math" in dataset:
    #     return parse_math_answer(text)
    # elif "countdown" in dataset:
    #     return parse_ctd_answer(text)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported for answer parsing.")

def eval_result(dataset, answer, ground_truth):
    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")
        
    if "svamp" in dataset or "gsm" in dataset:
        if answer is None:
            return False
        return float(answer) == float(ground_truth)
    elif "math" in dataset:
        if answer is None:
            return False
        return is_equiv(answer, ground_truth)
    elif "countdown" in dataset:
        is_valid = validate_equation(answer, ground_truth[0])  
        if is_valid:
            eval_result = evaluate_equation(answer)
            if abs(eval_result - ground_truth[1]) < 1e-5:
                return True
    return False

def get_acc(directory):
    if not directory:
        raise ValueError("Directory path cannot be empty.")
    
    if not glob(f"{directory}/*.json"):
        raise FileNotFoundError(f"No JSON files found in the directory: {directory}")
    model = directory.split("/")[-2]
    folder = directory.split("/")[-1]
    dataset = folder.split("_")[0]
    generate_length = folder.split("_")[2]
    total_steps = folder.split("_")[4]
    temperature = folder.split("_")[6]
    vote = "vote" in folder

    print(f"evaluating {dataset} with {model} on {generate_length} length, {total_steps} total steps, {temperature} temperature, vote: {vote}")

    json_files = glob(f"{directory}/*.json", recursive=True)
    json_files.sort()

    final_correct_questions = 0
    vote_correct_questions = 0
    total_questions = 0

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        generations = data["generations"]

        if vote:
            for generation in generations:
                vote_answer = generation["vote_answer"]
                final_answer = generation["final_answer"]
                ground_truth = generation["ground_truth"]
                vote_correct = eval_result(dataset, vote_answer, ground_truth)
                final_correct = eval_result(dataset, final_answer, ground_truth)
                final_correct_questions += final_correct
                vote_correct_questions += vote_correct
                # if final_correct and not vote_correct:
                #     print(f"Final answer {final_answer} is correct, but vote answer {vote_answer} is not correct.")   
                total_questions += 1
        else:
            for generation in generations:
                generated_text = generation["generations"]
                final_answer = parse_answer(dataset, generated_text)
                ground_truth = generation["ground_truth"]
                if eval_result(dataset, final_answer, ground_truth):
                    final_correct_questions += 1
                total_questions += 1

    if vote:
        final_accuracy = final_correct_questions / total_questions * 100
        vote_accuracy = vote_correct_questions / total_questions * 100
        print(f"total process questions: {total_questions}")
        print(f"Final answer accuracy: {final_accuracy:.2f}%")
        print(f"Vote answer accuracy: {vote_accuracy:.2f}%")
    else:
        final_accuracy = final_correct_questions / total_questions * 100
        print(f"total process questions: {total_questions}")
        print(f"Final answer accuracy: {final_accuracy:.2f}%")

def get_acc_all():
    datasets = [
        "gsm8k",
        "math", 
        "countdown",
        "svamp",
    ]
    lengths = [
        128, 
        256,
        512
    ]

    for dataset in datasets:
        for length in lengths:
            steps = length // 2
            directory = f"outputs/LLaDA-8B-Instruct/{dataset}_gen_{length}_steps_{steps}_temp_0.0_vote"
            get_acc(directory)

def main():
    # directory = "eval/outputs/LLaDA-8B-Instruct/gsm8k_gen_256_steps_256_temp_0.0test_N2E10"
    directory = "eval/outputs/LLaDA-8B-Instruct/gsm8k_gen_128_steps_128_temp_0.0test"
    get_acc(directory)

if __name__ == "__main__":
    main()
    # get_acc_all()
    