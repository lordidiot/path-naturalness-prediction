import re
import sys
import json
import pickle
from pathlib import Path as _Path
from openai import AsyncOpenAI, OpenAI
from pprint import pprint
import uuid

from .types import Path
from .one_shot import OneShotPrompting
from .get_data import QUESTION_TEMPLATE, ANSWER_PATTERN

def get_query_request_data(path_a: Path, path_b: Path) -> dict:
    prompt = QUESTION_TEMPLATE.format(A=path_a.short, B=path_b.short)
    data = {
        "url": "/v1/chat/completions",
        "custom_id": path_a.id + '_' + path_b.id + '_' + uuid.uuid4().hex[:16],
        "method": "POST",
        "body": {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4o-mini",
            "max_tokens": 2000,
        }
    }
    return data

def convert_batch_output_to_answers(batch_output_path: _Path, answers_path: _Path):
    with open(batch_output_path, 'r') as batch:
        with open(answers_path, 'w') as out:
            for line in batch:
                response = json.loads(line)
                path_a_id, path_b_id, _ = response['custom_id'].split('_')
                # choice = json.loads(response['response']['body']['choices'][0]['message']['content'])['choice']
                match = re.search(ANSWER_PATTERN, response['response']['body']['choices'][0]['message']['content'])
                choice = match.group(1) if match else None
                out.write(f"{path_a_id}_{path_b_id}_{path_a_id if choice == 'A' else path_b_id}\n")

def get_paths_from_answer(answer: str, paths: dict[str, any]) -> tuple[Path, Path]:
    a_id, b_id, _ = answer.strip().split('_')
    path_a = Path(
        id=a_id,
        short=paths[a_id[:-1]]['forward' if a_id[-1] == 'f' else 'reverse']['short']
    )
    path_b = Path(
        id=b_id,
        short=paths[b_id[:-1]]['forward' if b_id[-1] == 'f' else 'reverse']['short']
    )
    return path_a, path_b

def main():
    if len(sys.argv) < 4:
        print("Usage: python -m llm.batch <paths.pkl> <answers.txt> <input_path>")
        return
    paths_path = _Path(sys.argv[1])
    answers_path = _Path(sys.argv[2])
    input_path = _Path(sys.argv[3])
    batch_input_file_id = None
    if len(sys.argv) >= 5:
        batch_input_file_id = sys.argv[4]

    with open(paths_path, 'rb') as f:
        paths = pickle.load(f)

    if input_path.exists():
        print(f"{input_path} already exists. Skipping generation.")
    else:
        print("Generating input file...")
        # prompter = OneShotPrompting(AsyncOpenAI())
        with open(answers_path, 'r') as answers:
            with open(input_path, 'w') as out:
                for line in answers:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    path_a, path_b = get_paths_from_answer(line, paths)
                    out.write(json.dumps(get_query_request_data(path_a, path_b)) + '\n')

    if True:
        client = OpenAI()
        if batch_input_file_id is None:
            with open(input_path, 'rb') as f:
                batch_input_file = client.files.create(file=f, purpose="batch")
                batch_input_file_id = batch_input_file.id
                print(f"Batch input file created: {batch_input_file}")

        job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "test batch job"
            }
        )
        print("Batch job:")
        pprint(job)


if __name__ == '__main__':
    main()