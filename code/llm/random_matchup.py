import json
import pickle
import random

from .types import Path
from .batch import get_query_request_data
from .pairwise_batch import submit_batch

def get_random(id: str, path_data) -> Path:
    dir = 'forward' if random.getrandbits(1) else 'reverse'
    short = path_data[dir]['short']
    return Path(id=id + dir[0], short=short)

def prepare_pairs(filename: str,
                  clip: tuple[int, int],
                  out: str,
                  against: int = 9,
                  ) -> list[tuple[Path, Path]]:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    keys = list(data.keys())[clip[0]:clip[1]]
    pairs: list[tuple[Path, Path]] = []
    for i, key in enumerate(keys):
        other_keys = random.sample(keys, against)
        while i in other_keys:
            other_keys = random.sample(keys, against)
        for other_key in other_keys:
            pairs.append((
                get_random(key, data[key]),
                get_random(other_key, data[other_key]),
            ))
    with open(out, 'w') as f:
        for a, b in pairs:
            f.write(json.dumps(get_query_request_data(a, b)) + '\n')

def run_science(clip: tuple[int, int], submit: bool):
    batch_input = "../data/science/batch_input.jsonl"
    prepare_pairs(filename="../data/science/paths.pkl",
                  out=batch_input,
                  clip=clip)
    if submit:
        submit_batch(batch_input, description="Science batch job on original")

def run_money(clip: tuple[int, int], submit: bool):
    batch_input = "../data/money/batch_input.jsonl"
    prepare_pairs(filename="../data/money/paths.pkl",
                  out=batch_input,
                  clip=clip)
    if submit:
        submit_batch(batch_input, description="Money batch job on original")

def main():
    # Dataset has ~2.8k rows, so pick range to be either (0, 1500) or (1500, 3000)
    clip = (0, 1500)
    # Change this to True to submit batch job to OpenAI
    submit = False
    # Comment out accordingly
    run_science(clip, submit)
    run_money(clip, submit)

    # After this, to parse the output, download it
    # and run the following directly in python REPL
    # from llm.batch import convert_batch_output_to_answers as f
    # f("<your file name>", "<output txt file name>")


if __name__ == '__main__':
    main()
