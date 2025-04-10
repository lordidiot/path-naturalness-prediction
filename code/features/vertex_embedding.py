import numpy as np

from .base_feature import BaseVertexFeature
from .generate import run_vertex_feature_on_original, enumerate_vertices, run_vertex_feature_on_fixed_endpoints, enumerate_fixed_endpoints_vertices

# <rant>The paper says they use the 840B version, but the data they
#       uploaded corresponds to the 42B version?!?!?!?! </rant>
GLOVE_PATH = '../data/glove.42B.300d.txt' # https://nlp.stanford.edu/data/glove.42B.300d.zip
FEATURE_NAME = 'v_enc_dim300'

class VertexEmbeddingFeature(BaseVertexFeature):
    def __init__(self, vertices: set[str]):
        self.glove_embeddings = self._cache_glove_embeddings(vertices)

    def _cache_glove_embeddings(self, words: set[str]):
        glove_embeddings = {}
        with open(GLOVE_PATH, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                if line[0] in words:
                    glove_embeddings[line[0]] = np.array(line[1:], dtype=np.float32)
        return glove_embeddings

    def calculate(self, vertex: str) -> np.ndarray:
        return self.glove_embeddings[vertex]


if __name__ == "__main__":
    science_data_path = "../data/science/paths.pkl"
    money_data_path = "../data/money/paths.pkl"
    science_vertices = enumerate_vertices(science_data_path)
    money_vertices = enumerate_vertices(money_data_path)
    feature = VertexEmbeddingFeature(science_vertices.union(money_vertices))
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=science_data_path,
                                   out=f"../data/science/features/{FEATURE_NAME}.pkl")
    run_vertex_feature_on_original(vertex_feature=feature,
                                   data_path=money_data_path,
                                   out=f"../data/money/features/{FEATURE_NAME}.pkl")

    fixed_endpoints_science_data_path = "../data/fixed_endpoints/science_paths_fixed_endpoints.pkl"
    fixed_endpoints_money_data_path = "../data/fixed_endpoints/money_paths_fixed_endpoints.pkl"
    fixed_endpoints_science_vertices = enumerate_fixed_endpoints_vertices(fixed_endpoints_science_data_path)
    fixed_endpoints_money_vertices = enumerate_fixed_endpoints_vertices(fixed_endpoints_money_data_path)
    fixed_endpoints_feature = VertexEmbeddingFeature(fixed_endpoints_money_vertices.union(fixed_endpoints_science_vertices))

    run_vertex_feature_on_fixed_endpoints(vertex_feature=fixed_endpoints_feature,
                                        data_path=fixed_endpoints_money_data_path,
                                        out=f"../data/fixed_endpoints/money_features/{FEATURE_NAME}.pkl")
    run_vertex_feature_on_fixed_endpoints(vertex_feature=fixed_endpoints_feature,
                                        data_path=fixed_endpoints_science_data_path,
                                        out=f"../data/fixed_endpoints/science_features/{FEATURE_NAME}.pkl")
