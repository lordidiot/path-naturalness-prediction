from tqdm import tqdm
import asyncio
import dotenv

from .types import Path, Answer
from .pairwise_batch import AnswerReader
from fixed_endpoints.utils import load_pickle

dotenv.load_dotenv()

class RoundRobin:
    def __init__(self, file: str):
        self.prompter = AnswerReader(file)

    async def run(self, paths: list[Path]) -> dict[Path, float]:
        scores = {path: 0.0 for path in paths}
        pairs = [(path_a, path_b) for i, path_a in enumerate(paths) for path_b in paths[i + 1:]]
        answers: list[Answer] = await asyncio.gather(*(self.prompter.query(path_a, path_b) for path_a, path_b in pairs))
        for (path_a, path_b), answer in zip(pairs, answers):
            if answer.choice == 'A':
                scores[path_a] += 1
            elif answer.choice == 'B':
                scores[path_b] += 1
        for path in paths:
            scores[path] /= len(paths) - 1
        return scores


async def run(data_file: str,
              answer_file: str,
              clips: list[tuple[int, int]],
              out: str):
    data = load_pickle(data_file)
    rr = RoundRobin(answer_file)
    scores: dict[Path, float] = {}
    for clip in clips:
        for _, paths in tqdm(list(data.items())[clip[0]:clip[1]]):
            paths = [Path(id=path.id, short=path.short()) for path in paths]
            score = await rr.run(paths)
            scores.update(score)
    with open(out, "w") as f:
        for path, score in scores.items():
            f.write(f"{path.id}_{score:.5f}\n")

async def run_science(clips: list[tuple[int, int]]):
    await run(data_file="../data/fixed_endpoints/science_paths_fixed_endpoints.pkl",
              answer_file="../data/fixed_endpoints/science_answer.txt",
              clips=clips,
              out="../data/fixed_endpoints/science_rr.txt")

async def run_money(clips: list[tuple[int, int]]):
    await run(data_file="../data/fixed_endpoints/money_paths_fixed_endpoints.pkl",
              answer_file="../data/fixed_endpoints/money_answer.txt",
              clips=clips,
              out="../data/fixed_endpoints/money_rr.txt")


if __name__ == "__main__":
    clips = [(0, 200), (1000, 1200)]
    asyncio.run(run_science(clips))
    asyncio.run(run_money(clips))
