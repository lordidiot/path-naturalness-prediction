import json
from openai import OpenAI
from pprint import pprint
from typing import Literal

from .types import Path, Prompting, Answer
from .batch import get_query_request_data
from fixed_endpoints.utils import load_pickle

def prepare(filename: str, clips: tuple[int, int], out: str) -> None:
    data = load_pickle(filename)
    pairs: list[tuple[Path, Path]] = []
    for clip in clips:
        for _, paths in list(data.items())[clip[0]:clip[1]]:
            paths = [Path(id=path.id, short=path.short()) for path in paths]
            for i, a in enumerate(paths):
                for b in paths[i + 1:]:
                    pairs.append((a, b))
    with open(out, 'w') as f:
        for a, b in pairs:
            f.write(json.dumps(get_query_request_data(a, b)) + '\n')

def submit_batch(file: str, description: str) -> None:
    client = OpenAI()
    with open(file, 'rb') as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
        batch_input_file_id = batch_input_file.id
        print(f"Batch input file created: {batch_input_file}")

    job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )
    print("Batch job:")
    pprint(job)

def run_science(clips: list[tuple[int, int]], submit: bool = True) -> None:
    batch_input = "../data/fixed_endpoints/science_batch_input.jsonl"
    prepare(filename="../data/fixed_endpoints/science_paths_fixed_endpoints.pkl",
            clips=clips,
            out=batch_input)
    if submit:
        submit_batch(batch_input, description="Science batch job")

def run_money(clips: list[tuple[int, int]], submit: bool = True) -> None:
    batch_input = "../data/fixed_endpoints/money_batch_input.jsonl"
    prepare(filename="../data/fixed_endpoints/money_paths_fixed_endpoints.pkl",
            clips=clips,
            out=batch_input)
    if submit:
        submit_batch(batch_input, description="Money batch job")

class AnswerReader(Prompting):
    def __init__(self, file: str):
        self.answers: dict[(str, str), Literal["A", "B"]] = {}
        with open(file) as f:
            for line in f:
                id1, id2, winner = line.strip().split("_")
                self.answers[id1, id2] = "A" if winner == id1 else "B"
    
    async def query(self, path_a: Path, path_b: Path):
        if (path_a.id, path_b.id) in self.answers:
            return Answer(path_a=path_a, path_b=path_b, choice=self.answers[path_a.id, path_b.id])
        elif (path_b.id, path_a.id) in self.answers:
            answer = "A" if self.answers[path_b.id, path_a.id] == "B" else "B"
            return Answer(path_a=path_a, path_b=path_b, choice=answer)
        else:
            raise ValueError(f"Pair ({path_a}, {path_b}) not found")

if __name__ == '__main__':
    # Ranges run: (0, 800), (1000, 1200)
    clips = [(600, 800)]
    submit = True
    run_science(clips, submit=submit)
    # run_money(clips, submit=submit)
