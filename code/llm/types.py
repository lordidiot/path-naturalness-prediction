from pydantic import BaseModel
from typing import Literal
import json
import logging

logger = logging.getLogger(__name__)

class Path(BaseModel):
    id: str  # eg: "38276f"
    short: str  # eg: "A <--RelatedTo--> B"

class Answer(BaseModel):
    path_a: Path
    path_b: Path
    choice: Literal["A", "B"]

    def __str__(self):
        return f"{self.path_a.id}_{self.path_b.id}_{self.choice}"

    @staticmethod
    def from_llm_response(response: str, path_a: Path, path_b: Path) -> 'Answer':
        response = response.strip("`json ")
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response}")
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data)} from response: {response}")
        if 'choice' not in data:
            raise ValueError(f"Expected 'choice' key in JSON response: {response}")
        if 'explanation' not in data:
            raise ValueError(f"Expected 'explanation' key in JSON response: {response}")
        logger.debug(f"Loaded JSON data: {data}, path_a: {path_a}, path_b: {path_b}")
        return Answer(path_a=path_a, path_b=path_b, choice=data['choice'])
