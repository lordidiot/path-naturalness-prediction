from openai import AsyncOpenAI
import asyncio
from tqdm import tqdm

from fixed_endpoints.utils import load_pickle
from .types import Path, Prompting
from .one_shot import OneShotPrompting

def get_pairs_done(filename: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with open(filename, "r") as f:
        for line in f.readlines():
            a, b, _ = line.split("_")
            pairs.add((a, b))
    return pairs

async def compare_paths(paths: list[Path],
                        out: str,
                        prompter: Prompting,
                        done_pairs: set[tuple[str, str]]) -> None:
    pairs = [(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]]
    with open(out, "a") as f:
        for a, b in pairs:
            if (a.id, b.id) in done_pairs:
                continue
            answer = await prompter.query(a, b)
            f.write(f"{str(answer)}\n")

async def run(filename: str,
              out: str,
              range: tuple[int, int],
              pairs: set[tuple[str, str]]) -> None:
    # Range: for the purpose of cost monitoring
    # pairs: to avoid repeating the same pairs
    data = load_pickle(filename)
    client = AsyncOpenAI()
    prompter = OneShotPrompting(client)
    tasks = []
    for _, paths in list(data.items())[range[0]:range[1]]:
        paths = [Path(id=path.id, short=path.short()) for path in paths]
        tasks.append(compare_paths(paths, out, prompter, pairs))
    await asyncio.gather(*tasks)

# Due to OpenAI rate limit, we can only do async 10 at a time
async def run_batch(filename: str,
                    out: str, clip: tuple[int, int],
                    interval: int,
                    pairs: set[tuple[str, str]]):
    start, end = clip
    for i in tqdm(list(range(start, end, interval))):
        r = (i, i + interval)
        await run(filename=filename,
                  out=out,
                  range=r,
                  pairs=pairs)

async def run_science(range: tuple[int, int], interval: int = 10):
    pairs = get_pairs_done("../data/fixed_endpoints/science_answers.txt")
    await run_batch(filename="../data/fixed_endpoints/science_paths_fixed_endpoints.pkl",
                    out="../data/fixed_endpoints/science_answers.txt",
                    clip=range,
                    interval=interval,
                    pairs=pairs)

async def run_money(range: tuple[int, int], interval: int = 10):
    pairs = get_pairs_done("../data/fixed_endpoints/money_answers.txt")
    await run_batch(filename="../data/fixed_endpoints/money_paths_fixed_endpoints.pkl",
                    out="../data/fixed_endpoints/money_answers.txt",
                    clip=range,
                    interval=interval,
                    pairs=pairs)


if __name__ == "__main__":
    # Ranges run: (0, 200)
    # Ranges half-run: (1000, 1200)
    task = run_science(range=(1000, 1200))
    asyncio.run(task)
