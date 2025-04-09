import pickle

from .base_feature import BaseEdgeFeature
from .generate import run_edge_feature_on_original, run_edge_feature_on_fixed_endpoints

class EdgeDirectionFeature(BaseEdgeFeature):
    def calculate(self, edge: tuple[str, str, str]) -> list[float]:
        _0, relation, _1 = edge
        forward = relation[-1] == ">"
        backward = relation[0] == "<"
        if forward and backward:
            return (0, 0, 1)
        elif forward:
            return (1, 0, 0)
        elif backward:
            return (0, 1, 0)
        else:
            raise ValueError(f"Invalid relation: {relation}")


def main():
    with open("../data/science/paths.pkl", "rb") as f:
        paths = pickle.load(f)
    with open("science/features/e_dir.pkl", "rb") as f:
        e_dir = pickle.load(f)
    key = list(e_dir.keys())[95]
    direction = 'forward' if key[-1] == 'f' else 'reverse'
    path = paths[key[:-1]][direction]['short']
    items = path.split(" ")
    edges = [(items[i], items[i + 1], items[i + 2]) for i in range(0, len(items) - 2, 2)]
    feature = EdgeDirectionFeature()
    expected = e_dir[key]
    actual = feature.calculate_batch(edges)
    print("expected", expected)
    print("actual", actual)


if __name__ == "__main__":
    # main()
    feature = EdgeDirectionFeature()
    run_edge_feature_on_original(feature,
                                 "../data/science/paths.pkl",
                                 "../data/science/features/e_dir.pkl")
    run_edge_feature_on_original(feature,
                                 "../data/money/paths.pkl",
                                 "../data/money/features/e_dir.pkl")
                                 
    run_edge_feature_on_fixed_endpoints(feature,
                                        "../data/fixed_endpoints/money_paths_fixed_endpoints.pkl",
                                        "../data/fixed_endpoints/money_features/e_dir.pkl")
    run_edge_feature_on_fixed_endpoints(feature,
                                        "../data/fixed_endpoints/science_paths_fixed_endpoints.pkl",
                                        "../data/fixed_endpoints/science_features/e_dir.pkl")
