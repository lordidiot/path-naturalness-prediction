from openai import AsyncOpenAI
import asyncio
import dotenv

from .types import Path, Answer, Prompting

dotenv.load_dotenv()

QUESTION_TEMPLATE = """
Which of the following paths connecting two concepts is the most natural?

A) {A}
B) {B}

Your response should be in JSON with the following format:
{{
    "choice": "A" or "B"
    "explanation": "Your explanation here"
}}
"""

class OneShotPrompting(Prompting):
    def __init__(self, client: AsyncOpenAI):
        self.client = client
    
    async def query(self, path_a: Path, path_b: Path) -> Answer:
        prompt = QUESTION_TEMPLATE.format(A=path_a.short, B=path_b.short)
        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content
        return Answer.from_llm_response(response, path_a, path_b)

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
