import pickle

from .base_feature import BaseVertexFeature
from .sense_score import SenseScore
from .generate import enumerate_vertices, enumerate_fixed_endpoints_vertices, run_vertex_feature_on_original, run_vertex_feature_on_fixed_endpoints

class EdgeSenseScore(SenseScore, BaseVertexFeature):
    def calculate_batch(self, vertices: list[str]) -> list[list[float]]:
        denom_maxs: list[tuple[float, float]] = []
        prev_maxs: list[float] = []
        next_maxs: list[float] = []
        for i in range(len(vertices)):
            prev_max = 0
            next_max = 0
            curr_max = (0, 0)
            # print(self.word_senses[vertices[i]])
            for sense in self.word_senses[vertices[i]]:
                if i > 0:
                    prev = self.sense_similarity(vertices[i - 1], sense)
                    prev_max = max(prev_max, prev)
                else:
                    prev = 0
                if i < len(vertices) - 1:
                    next = self.sense_similarity(vertices[i + 1], sense)
                    next_max = max(next_max, next)
                else:
                    next = 0
                if prev + next > curr_max[0] + curr_max[1]:
                    curr_max = (prev, next)
            denom_maxs.append(curr_max)
            prev_maxs.append(prev_max)
            next_maxs.append(next_max)
        scores: list[list[float]] = []
        for i in range(len(vertices) - 1):
            score = (denom_maxs[i + 1][0] + denom_maxs[i][1]) / (prev_maxs[i + 1] + next_maxs[i])
            scores.append([score])
        return scores


def main():
    vertices = enumerate_vertices("../data/science/paths.pkl")
    feature = EdgeSenseScore(vertices)
    with open("../data/science/paths.pkl", "rb") as f:
        paths = pickle.load(f)
    with open("science/features/e_sense.pkl", "rb") as f:
        e_sense = pickle.load(f, encoding="latin1")
    for key in list(e_sense.keys())[:10]:
        direction = 'forward' if key[-1] == 'f' else 'reverse'
        path = paths[key[:-1]][direction]['short']
        words = [item.lower() for i, item in enumerate(path.split(" ")) if i % 2 == 0]
        expected = e_sense[key]
        actual = feature.calculate_batch(words)
        print("expected", expected)
        print("actual", actual)


if __name__ == '__main__':
    # main()
    science_data_path = "../data/science/paths.pkl"
    money_data_path = "../data/money/paths.pkl"
    science_vertices = enumerate_vertices(science_data_path)
    money_vertices = enumerate_vertices(money_data_path)
    feature = EdgeSenseScore(science_vertices.union(money_vertices))

    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=science_data_path,
                                   out="../data/science/features/e_sense.pkl")
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=money_data_path,
                                   out="../data/money/features/e_sense.pkl")

    fixed_endpoints_science_data_path = "../data/fixed_endpoints/science_paths_fixed_endpoints.pkl"
    fixed_endpoints_money_data_path = "../data/fixed_endpoints/money_paths_fixed_endpoints.pkl"
    fixed_endpoints_science_vertices = enumerate_fixed_endpoints_vertices(fixed_endpoints_science_data_path)
    fixed_endpoints_money_vertices = enumerate_fixed_endpoints_vertices(fixed_endpoints_money_data_path)
    fixed_endpoints_feature = EdgeSenseScore(fixed_endpoints_money_vertices.union(fixed_endpoints_science_vertices))

    run_vertex_feature_on_fixed_endpoints(vertex_feature=fixed_endpoints_feature,
                                        data_path=fixed_endpoints_money_data_path,
                                        out="../data/fixed_endpoints/money_features/e_sense.pkl")
    run_vertex_feature_on_fixed_endpoints(vertex_feature=fixed_endpoints_feature,
                                        data_path=fixed_endpoints_science_data_path,
                                        out="../data/fixed_endpoints/science_features/e_sense.pkl")
