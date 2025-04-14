from tqdm import tqdm
import dotenv
import random
import pickle
from typing import Literal

from .types import Path
from .finegrained import extract_contents_from_jsonl

dotenv.load_dotenv()

class EloRanking:
    def __init__(self, k: float = 32):
        self.k = k
    
    def run(self,
                  paths: list[Path],
                  raw_results: list[tuple[tuple[Path, Path], list[Literal["A", "B"]]]],
                  epochs: int = 10000,
                  matches_per_epoch: int = 100,
                  threshold: int = 10,
                  out: str = "../data/science/elo_ranking_softlabel.txt",
                  ) -> dict[Path, float]:
        scores = {path: 1500 for path in paths}
        for _ in tqdm(range(epochs)):
            q = {path: self._q(scores[path]) for path in paths}
            matches = random.sample(raw_results, matches_per_epoch)
            answers = list(map(
                lambda result: result[1],
                matches
            ))
            pairs = list(map(
                lambda result: result[0],
                matches,
            ))

            max_difference = 0
            for (path_a, path_b), outcomes in zip(pairs, answers):
                a_expected = round(q[path_a] / (q[path_a] + q[path_b]), 2)
                b_expected = round(q[path_b] / (q[path_a] + q[path_b]), 2)
                answer = random.choice(outcomes)
                if answer == 'A':
                    a_difference = round(self.k * (1 - a_expected))
                    b_difference = round(self.k * (-b_expected))
                elif answer == 'B':
                    a_difference = round(self.k * (-a_expected))
                    b_difference = round(self.k * (1 - b_expected))
                scores[path_a] += a_difference
                scores[path_b] += b_difference
                max_difference = max(max_difference, abs(a_difference), abs(b_difference))
            
            if max_difference < threshold:
                break
        
        with open(out, "w") as f:
            for path, score in scores.items():
                f.write(f"{path.id}_{score}\n")
    
    def _q(self, rating: float) -> float:
        return 10 ** (rating / 400)

def run_original(path_file: str = "../data/science/paths.pkl",
                  base_answers: str = "../data/science/llm_answers_2.txt",
                  jsonl_file: str = "../data/finegrained/original/{n}_output.jsonl",
                  jsonl_range: tuple[int, int] = (1, 30)):
    with open(path_file, "rb") as f:
        data = pickle.load(f)
    paths: list[Path] = []
    for key, d in data.items():
        paths.append(Path(id=key + "f", short=d["forward"]["short"]))
        paths.append(Path(id=key + "r", short=d["reverse"]["short"]))
    raw_results: list[tuple[tuple[Path, Path], list[Literal["A", "B"]]]] = []
    with open(base_answers) as f:
        for line in f:
            a_id, b_id, _ = line.strip().split("_")
            a_short = data[a_id[:-1]]["forward" if a_id[-1] == "f" else "reverse"]["short"]
            b_short = data[b_id[:-1]]["forward" if b_id[-1] == "f" else "reverse"]["short"]
            path_a = Path(id=a_id, short=a_short)
            path_b = Path(id=b_id, short=b_short)
            raw_results.append(((path_a, path_b), []))
    start, end = jsonl_range
    for n in range(start, end + 1):
        labels: list[str] = extract_contents_from_jsonl(jsonl_file.format(n=n))
        for i, label in enumerate(labels):
            label = label.strip("$")
            raw_results[i][1].append(label)
    evaluator = EloRanking()
    evaluator.run(paths, raw_results)


if __name__ == "__main__":
    run_original()
