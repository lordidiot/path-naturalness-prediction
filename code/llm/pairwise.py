from openai import AsyncOpenAI
import asyncio

from fixed_endpoints.utils import load_pickle
from .types import Path, Prompting
from .one_shot import OneShotPrompting

async def compare_paths(paths: list[Path], out: str, prompter: Prompting) -> None:
    pairs = [(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]]
    with open(out, "a") as f:
        for a, b in pairs:
            answer = await prompter.query(a, b)
            print("Answer:", answer, flush=True)
            f.write(f"{str(answer)}\n")

async def main(filename: str, out: str, range: tuple[int, int]):
    # Range: for the purpose of cost monitoring
    data = load_pickle(filename)
    client = AsyncOpenAI()
    prompter = OneShotPrompting(client)
    tasks = []
    for _, paths in list(data.items())[range[0]:range[1]]:
        paths = [Path(id=path.id, short=path.short()) for path in paths]
        tasks.append(compare_paths(paths, out, prompter))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    task = main(filename="../data/fixed_endpoints/science_paths_fixed_endpoints.pkl",
                out="../data/fixed_endpoints/science_answers.txt",
                range=(0, 10))
    asyncio.run(task)
