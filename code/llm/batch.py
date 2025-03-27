import sys
import json
import pickle
from pathlib import Path as _Path
from openai import AsyncOpenAI, OpenAI
from pprint import pprint

from .types import Path
from .one_shot import OneShotPrompting

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
        prompter = OneShotPrompting(AsyncOpenAI())
        with open(answers_path, 'r') as answers:
            with open(input_path, 'w') as out:
                for line in answers:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    path_a, path_b = get_paths_from_answer(line, paths)
                    out.write(json.dumps(prompter.get_query_request_data(path_a, path_b)) + '\n')
    
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