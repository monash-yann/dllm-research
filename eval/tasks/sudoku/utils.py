import logging
import signal
import datasets


eval_logger = logging.getLogger(__name__)


# taken from
def doc_to_text(doc: dict) -> str:
    return f"Puzzle:\n{doc['puzzle']}\n\nSolution:\n"

def doc_to_target(doc: dict) -> str:
    return doc["solution"].strip()

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        puzzle_str = str(doc["Puzzle"])
        if len(puzzle_str) < 16:
            puzzle_str = puzzle_str.rjust(16, '0')
        puzzle_str = '\n'.join(puzzle_str[i: i+4] for i in range(0, len(puzzle_str), 4))
        out_doc = {
            "puzzle": puzzle_str.strip(),
            "solution": str(doc["Solution"]).strip(),
            "system_prompt": SUDOKU_SYSTEM_PROMPT,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = 1
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "puzzle": "4100\n0001\n1300\n2000",
            "solution": "4132\n3241\n1324\n2413",
            "few_shot": 1,
        },
        {
            "puzzle": "0004\n0321\n0203\n3002",
            "solution": "2134\n4321\n1243\n3412",
            "few_shot": 1,
        },
        {
            "puzzle": "4123\n0000\n0402\n2300",
            "solution": "4123\n3214\n1432\n2341",
            "few_shot": 1,
        },
        {
            "puzzle": "1432\n0041\n3000\n4000",
            "solution": "1432\n2341\n3214\n4123",
            "few_shot": 1,
        },
        {
            "puzzle": "0020\n0341\n0210\n1002",
            "solution": "4123\n2341\n3214\n1432",
            "few_shot": 1,
        },
    ]


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    candidate = results[0]

    true_solution = doc["solution"].strip().replace("\n", "")
    pred_solution = candidate.strip().replace("\n", "")

    print(f"pred_answer[{pred_solution}]==true_answer[{true_solution}]: {pred_solution == true_solution}")

    res = {
        "exact_match": true_solution == pred_solution,
    }
    return res


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



SUDOKU_SYSTEM_PROMPT = """
Solve this 4x4 Sudoku puzzle represented as a 16-digit string (read left-to-right, top-to-bottom) where '0'=empty cell.

Requirements:
1. Replace ALL '0's with digits 1-4
2. Follow STRICT Sudoku rules:
   - Rows: Each must contain 1-4 exactly once
   - Columns: Each must contain 1-4 exactly once
   - 2x2 Boxes: Each must contain 1-4 exactly once
"""
# 3. Format answer as:
# <answer>
# [16-digit solution]
# </answer>

SUDOKU_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Here are some examples:
"""

