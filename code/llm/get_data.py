#!/usr/bin/env python3

import re
import os
import sys
import pickle
from dataclasses import dataclass
from openai import OpenAI
from openai.types.chat import ChatCompletion

@dataclass
class Path:
    id: str
    text: str
    short: str

    @staticmethod
    def from_id(path_id: str, paths: dict[str, any]) -> 'Path':
        path_code, direction = path_id[:-1], path_id[-1]
        path = paths[path_code]['forward' if direction == 'f' else 'reverse']
        return Path(path_id, path['text'], path['short'])

@dataclass
class Answer:
    path_a: Path
    path_b: Path
    choice: str | None

    @staticmethod
    def from_str(answer_str: str, paths: dict[str, any]) -> 'Answer':
        a, b, choice_id = answer_str.strip().split('_')
        path_a = Path.from_id(a, paths)
        path_b = Path.from_id(b, paths)
        choice = 'A' if choice_id == a else 'B'
        return Answer(path_a, path_b, choice)

    def __str__(self):
        choice = None if self.choice is None else (self.path_a.id if self.choice == 'A' else self.path_b.id)
        return f"{self.path_a.id}_{self.path_b.id}_{choice}"

QUESTION_TEMPLATE = """\
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of AB.

Which of the following paths connecting two concepts is the most natural?

A) {A}
B) {B}
""".strip()

ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([AB])\$?"

def get_response(client: OpenAI, answer: Answer) -> ChatCompletion:
    question_data = {
        "A": answer.path_a.short,
        "B": answer.path_b.short,
    }

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "user", "content": QUESTION_TEMPLATE.format(**question_data)},
        ]
    )
    match = re.search(ANSWER_PATTERN, completion.choices[0].message.content)
    choice = match.group(1) if match else None
    return Answer(answer.path_a, answer.path_b, choice)

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <answers.txt> <paths.pkl> <out_dir>")
        return
    answers_path, paths_path, out_dir = sys.argv[1:4]

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        print(f'Output directory "{out_dir}" is not a valid directory.')
        return

    with open(paths_path, 'rb') as f:
        paths = pickle.load(f)

    answers = []
    with open(answers_path, 'r') as f:
        for line in f.read().strip().split('\n'):
            answers.append(Answer.from_str(line, paths))

    client = OpenAI()

    print(answers[0])
    print(get_response(client, answers[0]))


if __name__ == '__main__':
    main()

