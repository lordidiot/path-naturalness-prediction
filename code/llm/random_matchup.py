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
                  against: int = 30,
                  batches: int = 10,
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
            # pairs.append((
            #     get_random(key, data[key]),
            #     get_random(other_key, data[other_key]),
            # ))
            
            # only use forward direction
            pairs.append((
                Path(id=f'{key}f', short=data[key]['forward']['short']),
                Path(id=f'{other_key}f', short=data[other_key]['forward']['short']),
            ))
    batch_size = len(pairs) // batches
    for i in range(batches):
        out_filename = f'{out}_{i}.jsonl'

        with open(out_filename, 'w') as f:
            for a, b in pairs[batch_size * i:batch_size * (i + 1)]:
                f.write(json.dumps(get_query_request_data(a, b)) + '\n')

def run_science(clip: tuple[int, int], submit: bool, against: int = 30, batches: int = 10):
    batch_input = "../data/science/batch_input"
    prepare_pairs(filename="../data/science/paths.pkl",
                  out=batch_input,
                  clip=clip,
                  against=against,
                  batches=batches)
    if submit:
        for i in range(batches):
            input_filename = f"{batch_input}_{i}.jsonl"
            submit_batch(input_filename, description="Science batch job on original")

def run_money(clip: tuple[int, int], submit: bool, against: int = 30, batches: int = 10):
    batch_input = "../data/money/batch_input"
    prepare_pairs(filename="../data/money/paths.pkl",
                  out=batch_input,
                  clip=clip,
                  against=against,
                  batches=batches)
    if submit:
        for i in range(batches):
            input_filename = f"{batch_input}_{i}.jsonl"
            submit_batch(input_filename, description="Money batch job on original")

def main():
    # Dataset has ~2.8k rows, (previously) pick range to be either (0, 1500) or (1500, 3000)
    # Upon second thought, this would restrict each batch's possible matchups to be half of the dataset
    # A better approach is to sample from the entire dataset
    # and then divide into different batches to submit
    clip = (0, 3000)
    # Change this to True to submit batch job to OpenAI
    submit = False
    # Comment out accordingly
    run_science(clip, submit, batches=5)
    # run_money(clip, submit)

    # After this, to parse the output, download it
    # and run the following directly in python REPL
    # from llm.batch import convert_batch_output_to_answers as f
    # f("<your file name>", "<output txt file name>")


if __name__ == '__main__':
    main()
