import pickle
from .types import Path

def load_pickle(filename) -> dict[tuple[str, str], list[Path]]:
    with open(filename, 'rb') as f:
        data: dict[tuple[str, str], list[Path]] = pickle.load(f)
        return data