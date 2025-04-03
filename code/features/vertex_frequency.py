from .base_feature import BaseVertexFeature
from .generate import run_vertex_feature_on_original, enumerate_vertices, run_vertex_feature_on_fixed_endpoints

GLOVE_PATH = '../data/glove.42B.300d.txt' # https://nlp.stanford.edu/data/glove.42B.300d.zip
FEATURE_NAME = 'v_freq_freq'

class VertexFrequencyFeature(BaseVertexFeature):
    def __init__(self, vertices: set[str]):
        self.glove_word_rankings = self._cache_glove_word_rankings(vertices)

    def _cache_glove_word_rankings(self, words: set[str]):
        glove_word_rankings = {}
        with open(GLOVE_PATH, 'r') as f:
            for i, line in enumerate(f):
                line = line.rstrip().split(' ')
                if line[0] in words:
                    glove_word_rankings[line[0]] = i + 1 # rank
        return glove_word_rankings

    def calculate(self, vertex: str) -> list[float]:
        return [1 / self.glove_word_rankings[vertex]]


if __name__ == "__main__":
    science_data_path = "../data/science/paths.pkl"
    money_data_path = "../data/money/paths.pkl"
    science_vertices = enumerate_vertices(science_data_path)
    money_vertices = enumerate_vertices(money_data_path)
    feature = VertexFrequencyFeature(science_vertices.union(money_vertices))
    '''
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=science_data_path,
                                   out=f"../data/science/features/{FEATURE_NAME}.pkl")
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=money_data_path,
                                   out=f"../data/money/features/{FEATURE_NAME}.pkl")
    '''
    run_vertex_feature_on_fixed_endpoints(feature,
                                        "../data/fixed_endpoints/money_paths_fixed_endpoints.pkl",
                                        f"../data/fixed_endpoints/money_features/{FEATURE_NAME}.pkl")
    run_vertex_feature_on_fixed_endpoints(feature,
                                        "../data/fixed_endpoints/science_paths_fixed_endpoints.pkl",
                                        f"../data/fixed_endpoints/science_features/{FEATURE_NAME}.pkl")


