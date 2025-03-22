from pydantic import BaseModel
from typing import Literal
import json
import logging

logger = logging.getLogger(__name__)

class Path(BaseModel):
    id: str  # eg: "38276f"
    short: str  # eg: "A <--RelatedTo--> B"

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Path):
            return False
        return self.id == other.id

class Answer(BaseModel):
    path_a: Path
    path_b: Path
    choice: Literal["A", "B"]

    def __str__(self):
        return f"{self.path_a.id}_{self.path_b.id}_{self.choice}"

class Prompting:
    async def query(self, path_a: Path, path_b: Path) -> Answer:
        raise NotImplementedError
