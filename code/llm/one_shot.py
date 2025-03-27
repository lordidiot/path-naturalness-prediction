from openai import AsyncOpenAI, OpenAI
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
    
    # Quite a hacky way to get the query data, but it works (I hope)
    def get_query_request_data(self, path_a: Path, path_b: Path) -> dict:
        prompt = QUESTION_TEMPLATE.format(A=path_a.short, B=path_b.short)
        client = OpenAI()
        data = {
            "url": "/v1/chat/completions",
            "custom_id": path_a.id + '_' + path_b.id,
            "method": "POST",
        }

        def hook_args(url: str, body: dict = {}, **kwargs):
            to_delete = []
            for key, value in body.items():
                if repr(value) == "NOT_GIVEN":
                    to_delete.append(key)
            for key in to_delete:
                del body[key]
            body["max_tokens"] = 2000
            data["body"] = body
        client.post = hook_args

        client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=Response,
        )
        return data

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
