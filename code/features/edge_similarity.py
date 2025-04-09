import numpy as np

from .base_feature import BaseEdgeFeature
from .generate import run_edge_feature_on_original, enumerate_vertices, run_edge_feature_on_fixed_endpoints, enumerate_fixed_endpoints_vertices 

GLOVE_PATH = '../data/glove.42B.300d.txt' # https://nlp.stanford.edu/data/glove.42B.300d.zip
FEATURE_NAME = 'e_vertexsim'

class EdgeSimilarityFeature(BaseEdgeFeature):
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

    def calculate(self, edge: tuple[str, str, str]) -> list[float]:
        v1, _, v2 = [i.lower() for i in edge]
        embed1 = self.glove_embeddings[v1]
        embed2 = self.glove_embeddings[v2]
        # paper calls this the cosine similarity, but it's actually the cosine distance
        distance = 1 - np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
        return [distance]


if __name__ == "__main__":
    science_data_path = "../data/science/paths.pkl"
    money_data_path = "../data/money/paths.pkl"
    science_vertices = enumerate_vertices(science_data_path)
    money_vertices = enumerate_vertices(money_data_path)
    feature = EdgeSimilarityFeature(science_vertices.union(money_vertices))
    run_edge_feature_on_original(edge_feature=feature,
                                 data_path=science_data_path,
                                 out=f"../data/science/features/{FEATURE_NAME}.pkl")
    run_edge_feature_on_original(edge_feature=feature,
                                 data_path=money_data_path,
                                 out=f"../data/money/features/{FEATURE_NAME}.pkl")

    fixed_endpoints_science_data_path = "../data/fixed_endpoints/science_paths_fixed_endpoints.pkl"
    fixed_endpoints_money_data_path = "../data/fixed_endpoints/money_paths_fixed_endpoints.pkl"
    fixed_endpoints_science_vertices = enumerate_fixed_endpoints_vertices(fixed_endpoints_science_data_path)
    fixed_endpoints_money_vertices = enumerate_fixed_endpoints_vertices(fixed_endpoints_money_data_path)
    fixed_endpoints_feature = EdgeSimilarityFeature(fixed_endpoints_money_vertices.union(fixed_endpoints_science_vertices))

    run_edge_feature_on_fixed_endpoints(edge_feature=fixed_endpoints_feature,
                                        data_path=fixed_endpoints_money_data_path,
                                        out=f"../data/fixed_endpoints/money_features/{FEATURE_NAME}.pkl")
    run_edge_feature_on_fixed_endpoints(edge_feature=fixed_endpoints_feature,
                                        data_path=fixed_endpoints_science_data_path,
                                        out=f"../data/fixed_endpoints/science_features/{FEATURE_NAME}.pkl")

