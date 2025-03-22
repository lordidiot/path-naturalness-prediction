from openai import AsyncOpenAI
import asyncio
import dotenv
from pydantic import BaseModel
from typing import Literal

from .types import Path, Answer, Prompting

dotenv.load_dotenv()

class Response(BaseModel):
    choice: Literal["A", "B"]
    explanation: str

    def to_answer(self, path_a: Path, path_b: Path) -> Answer:
        return Answer(path_a=path_a, path_b=path_b, choice=self.choice)

QUESTION_TEMPLATE = """
Which of the following paths connecting two concepts is the most natural?

A) {A}
B) {B}
"""

class OneShotPrompting(Prompting):
    def __init__(self, client: AsyncOpenAI):
        self.client = client
    
    async def query(self, path_a: Path, path_b: Path) -> Answer:
        prompt = QUESTION_TEMPLATE.format(A=path_a.short, B=path_b.short)
        completion = await self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=Response,
        )
        response = completion.choices[0].message.parsed
        return response.to_answer(path_a, path_b)

async def main():
    path_a = Path(
        id="1",
        short="Lead <--Synonym--> Take <--DistinctFrom--> Give <--RelatedTo--> Poison",
    )
    path_b = Path(
        id="2",
        short="Lead <--HasProperty--> Toxic <--RelatedTo--> Lethal <--RelatedTo--> Poison",
    )
    client = AsyncOpenAI()
    prompter = OneShotPrompting(client)
    answer = await prompter.query(path_a, path_b)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
