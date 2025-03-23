from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import asyncio
import dotenv

from .types import Path, Answer, Prompting
from .one_shot import OneShotPrompting

dotenv.load_dotenv()

class RoundRobin:
    def __init__(self, prompter: Prompting):
        self.prompter = prompter

    async def run(self, paths: list[Path]) -> dict[Path, float]:
        scores = {path: 0.0 for path in paths}
        pairs = [(path_a, path_b) for path_a in paths for path_b in paths if path_a != path_b]
        answers: list[Answer] = await tqdm.gather(*(self.prompter.query(path_a, path_b) for path_a, path_b in pairs))
        for (path_a, path_b), answer in zip(pairs, answers):
            if answer.choice == 'A':
                scores[path_a] += 1
            elif answer.choice == 'B':
                scores[path_b] += 1
        for path in paths:
            scores[path] /= len(paths) - 1
        return scores


async def main():
    paths = [
        Path(
            id="1",
            short="Lead <--Synonym--> Take <--DistinctFrom--> Give <--RelatedTo--> Poison",
        ),
        Path(
            id="2",
            short="Lead <--HasProperty--> Toxic <--RelatedTo--> Lethal <--RelatedTo--> Poison",
        ),
        Path(
            id="3",
            short="Purse --AtLocation--> House <--RelatedTo--> Type <--RelatedTo--> Unit",
        ),
        Path(
            id="4",
            short="Molecule --IsA--> Unit <--RelatedTo--> Day <--RelatedTo--> Holiday",
        )
    ]
    client = AsyncOpenAI()
    prompter = OneShotPrompting(client)
    rr = RoundRobin(prompter)
    scores = await rr.run(paths)
    print(scores)


if __name__ == "__main__":
    asyncio.run(main())
