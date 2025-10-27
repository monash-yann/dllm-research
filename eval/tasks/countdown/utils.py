import logging
import re
import signal
import string

import datasets


eval_logger = logging.getLogger(__name__)


# taken from
def doc_to_text(doc: dict) -> str:
    return f"Question: {doc['question']}\nSolution:"

def doc_to_target(doc: dict) -> str:
    return doc["answer"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        print(f"===> Processing doc: {doc}")
        out_doc = {
            "question": doc["input"].strip(),
            "answer": doc["output"].strip(),
            "system_prompt": COUNTDOWN_SYSTEM_PROMPT,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = 1
        return out_doc
    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "question": "15,44,79,50",
            "answer": "Let's try to combine 15 and 44. 44 - 15 = 29. Now we have 29 and the remaining number 79. We need to reach the target 50. Let's try 79 - 29 = 50. This works. The answer is: 44-15=29,79-29=50",
            "few_shot": 1,
        },
        {
            "question": "1,2,12,25",
            "answer": "We have 1, 2, 12 and the target is 25. Let's try multiplying 2 and 12. 2 * 12 = 24. Now we have 24 and the remaining number 1. We need to reach 25. 24 + 1 = 25. This is correct. The answer is: 2*12=24,1+24=25",
            "few_shot": 1,
        },
        {
            "question": "3,85,5,30",
            "answer": "The numbers are 3, 85, 5 and the target is 30. Let's try adding 85 and 5. 85 + 5 = 90. Now we have 90 and the remaining number 3. We need to reach 30. 90 / 3 = 30. That's the target. The answer is: 85+5=90,90/3=30",
            "few_shot": 1,
        },
    ]


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    pred_solution = results[0]
    pred_answer = extract_answer(pred_solution)
    true_answer = doc["answer"]

    print(f"pred_answer[{pred_answer}]==true_answer[{true_answer}]: {pred_answer == true_answer}")

    res = {
        "exact_match": true_answer == pred_answer,
    }
    return res


def extract_answer(text: str) -> str | None:
    match = re.search(r"The answer is:\s*(.*)", text, re.IGNORECASE)
    if match:
        result = match.group(1)
        result = result.strip().strip(string.punctuation)   # strip whitespace and punctuation
        return result
    return ''

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)



COUNTDOWN_SYSTEM_PROMPT = '''
For the given numbers, find a sequence of arithmetic operations that results in the target number.
Show your reasoning and conclude with "The answer is: [formula].
'''