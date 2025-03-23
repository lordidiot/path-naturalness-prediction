import requests
import pickle

from .base_feature import BaseVertexFeature, bidirectional_relations, unidirectional_relations
from .generate import run_vertex_feature_on_original

class VertexFrequencyFeature(BaseVertexFeature):
    def __init__(self):
        self.cache: dict[str, int] = dict()

    def calculate(self, vertex: str) -> list[float]:
        if vertex in self.cache:
            return [self.cache[vertex]]
        offset = 0
        count = 0
        edges = set()
        should_continue = True
        while should_continue:
            should_continue, count_incr = self._get_count(vertex, offset, edges)
            count += count_incr
            offset += 1000
        self.cache[vertex] = count
        return [count]
    
    def _get_count(self, vertex: str, offset: int, edges: set[tuple[str, str, str]]) -> tuple[bool, int]:
        obj = requests.get(f"http://api.conceptnet.io/c/en/{vertex}?offset={offset}&limit=1000").json()
        count = 0
        should_continue = len(obj["edges"]) > 0
        for edge in obj["edges"]:
            start = edge["start"]["@id"].strip("/").split("/")
            end = edge["end"]["@id"].strip("/").split("/")
            relation = edge["rel"]["@id"].strip("/").split("/")[1]
            if relation in bidirectional_relations:
                if (start[2], relation, end[2]) in edges or (end[2], relation, start[2]) in edges:
                    continue
            elif relation in unidirectional_relations:
                if (start[2], relation, end[2]) in edges:
                    continue
            else:
                continue
            edges.add((start[2], relation, end[2]))
            if start[0] != "c" or end[0] != "c":
                continue
            if start[1] != "en" or end[1] != "en":
                continue
            count += 1
        return should_continue, count


def main():
    feature = VertexFrequencyFeature()
    with open("../data/science/paths.pkl", "rb") as f:
        paths = pickle.load(f)
    with open("science/features/v_deg.pkl", "rb") as f:
        v_deg = pickle.load(f)
    key = list(v_deg.keys())[5]
    direction = 'forward' if key[-1] == 'f' else 'reverse'
    path = paths[key[:-1]][direction]['short']
    words = [item.lower() for i, item in enumerate(path.split(" ")) if i % 2 == 0]
    expected = v_deg[key]
    actual = feature.calculate_batch(words)
    print("expected", expected)
    print("actual", actual)

if __name__ == "__main__":
    # main()
    feature = VertexFrequencyFeature()
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path="../data/science/paths.pkl",
                                   out="../data/science/features/v_freq.pkl")
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path="../data/money/paths.pkl",
                                   out="../data/money/features/v_freq.pkl")
