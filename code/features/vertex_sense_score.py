import pickle

from .base_feature import BaseVertexFeature
from .sense_score import SenseScore
from .generate import enumerate_vertices, run_vertex_feature_on_original

class VertexSenseScore(SenseScore, BaseVertexFeature):
    def calculate_batch(self, vertices: list[str]) -> list[list[float]]:
        scores: list[list[float]] = []
        for i in range(len(vertices)):
            if i == 0 or i == len(vertices) - 1:
                scores.append([1])
                continue
            prev_max = 0
            next_max = 0
            curr_max = 0
            for sense in self.word_senses[vertices[i]]:
                prev = self.sense_similarity(vertices[i - 1], sense)
                prev_max = max(prev_max, prev)
                next = self.sense_similarity(vertices[i + 1], sense)
                next_max = max(next_max, next)
                curr_max = max(curr_max, prev + next)
            scores.append([curr_max / (prev_max + next_max)])
        return scores


def main():
    vertices = enumerate_vertices("../data/science/paths.pkl")
    feature = VertexSenseScore(vertices)
    with open("../data/science/paths.pkl", "rb") as f:
        paths = pickle.load(f)
    with open("science/features/v_sense.pkl", "rb") as f:
        v_sense = pickle.load(f, encoding='latin1')
    for key in list(v_sense.keys())[:10]:
        direction = 'forward' if key[-1] == 'f' else 'reverse'
        path = paths[key[:-1]][direction]['short']
        words = [item.lower() for i, item in enumerate(path.split(" ")) if i % 2 == 0]
        expected = v_sense[key]
        actual = feature.calculate_batch(words)
        print("expected", expected)
        print("actual", actual)
    # feature = VertexSenseScore(
    #     set(["knowledge", "book", "notebook", "restaurant", "bank", "money", "river", "deposit"]))
    # print(feature.calculate_batch(["knowledge", "book", "restaurant"]))
    # print(feature.calculate_batch(["knowledge", "book", "notebook"]))
    # print(feature.calculate_batch(["money", "bank", "river"]))
    # print(feature.calculate_batch(["money", "bank", "deposit"]))


if __name__ == '__main__':
    # main()
    science_data_path = "../data/science/paths.pkl"
    money_data_path = "../data/money/paths.pkl"
    science_vertices = enumerate_vertices(science_data_path)
    money_vertices = enumerate_vertices(money_data_path)
    feature = VertexSenseScore(science_vertices.union(money_vertices))
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=science_data_path,
                                   out="../data/science/features/v_sense.pkl")
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=money_data_path,
                                   out="../data/money/features/v_sense.pkl")
