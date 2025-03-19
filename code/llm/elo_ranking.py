from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
import dotenv
import random

from .types import Path, Answer, Prompting
from .one_shot import OneShotPrompting

dotenv.load_dotenv()

class EloRanking:
    def __init__(self, prompter: Prompting, k: float = 32):
        self.prompter = prompter
        self.k = k
    
    async def run(self,
                  paths: list[Path],
                  epochs: int = 100,
                  matches_per_epoch: int = 10,
                  threshold: int = 10) -> dict[Path, float]:
        scores = {path: 1500 for path in paths}
        for _ in tqdm(range(epochs)):
            q = {path: self._q(scores[path]) for path in paths}
            pairs = [(path_a, path_b) for path_a in paths for path_b in paths if path_a != path_b]
            pairs = random.sample(pairs, matches_per_epoch)

            answers: list[Answer] = await asyncio.gather(*(self.prompter.query(path_a, path_b) for path_a, path_b in pairs))

            expected_wins = {path: 0 for path in paths}
            for path_a, path_b in pairs:
                expected_wins[path_a] += round(q[path_a] / (q[path_a] + q[path_b]), 2)

            actual_wins = {path: 0 for path in paths}
            for (path_a, path_b), answer in zip(pairs, answers):
                if answer.choice == 'A':
                    actual_wins[path_a] += 1
                elif answer.choice == 'B':
                    actual_wins[path_b] += 1
            
            max_difference = 0
            for path in paths:
                difference = round(self.k * (actual_wins[path] - expected_wins[path]))
                scores[path] += difference
                max_difference = max(max_difference, abs(difference))
            
            if max_difference < threshold:
                break
        
        return scores
    
    def _q(self, rating: float) -> float:
        return 10 ** (rating / 400)


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
    er = EloRanking(prompter)
    scores = await er.run(paths,
                          epochs=10,
                          matches_per_epoch=5)
    print(scores)


if __name__ == "__main__":
    asyncio.run(main())
