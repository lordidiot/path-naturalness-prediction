import requests
from typing import Optional

from .base_feature import BaseEdgeFeature
from .generate import run_edge_feature_on_original, run_edge_feature_on_fixed_endpoints

FEATURE_NAME = 'e_weightsource'
CONCEPTNET_API_URL = "https://api.conceptnet.io"

class EdgeProvenanceFeature(BaseEdgeFeature):
    def __init__(self):
        self.cache = {}

    def _edge_id(self, edge) -> tuple[str, str, str]:
        a, rel, b = edge
        a, b = a.lower(), b.lower()
        rel = rel.split('-')[2]
        return tuple(sorted((a.lower(), b.lower()))) + (rel,)

    def _get_conceptnet_edge_data(self, edge: tuple[str, str, str]) -> Optional[dict]:
        a, rel, b = edge
        a, b = a.lower(), b.lower()
        rel = rel.split('-')[2]
        edge_id = self._edge_id(edge)
        if edge_id in self.cache:
            return self.cache[edge_id]
        data = requests.get(f"{CONCEPTNET_API_URL}/query?node=/c/en/{a}&other=/c/en/{b}&rel=/r/{rel}").json()
        self.cache[edge_id] = data
        return data

    def _get_sources(self, edge) -> list[str]:
        sources = []
        for source in edge['sources']:
            contributor = source['contributor']
            if contributor.startswith("/s/contributor/omcs"):
                contributor = "/s/contributor/omcs"
            elif contributor.startswith("/s/resource/dbpedia"):
                contributor = "/s/resource/dbpedia"
            elif contributor.startswith("/s/resource/wiktionary"):
                contributor = "/s/resource/wiktionary"
            sources.append(contributor)
        return list(set(sources))

    def calculate(self, edge: tuple[str, str, str]) -> list[float]:
        edge_data = self._get_conceptnet_edge_data(edge)
        if not len(edge_data['edges']): # some edges are not found in ConceptNet 5.8
            print("Edge not found:", edge)
            return [0, 0, 0, 0, 0, 0]
        e = edge_data['edges'][-1]
        sources = self._get_sources(e)
        weight = [0, 0, 0, 0, 0, 0]
        common_weights = [0.5, 1.0, 1.0, 0, 1.0, 2.0]
        all_sources = ['/s/resource/dbpedia', '/s/contributor/omcs', '/s/resource/opencyc/2012', '/s/resource/verbosity', '/s/resource/wiktionary', '/s/resource/wordnet/rdf/3.1']
        if len(sources) == 1:
            weight[all_sources.index(sources[0])] = e['weight']
        elif set(sources) == set(['/s/resource/verbosity', '/s/resource/wiktionary']):
            weight[4] = common_weights[4]
            weight[3] = e['weight'] - common_weights[4]
        else:
            print(f"Can't handle combination: {edge}, {sources}")
        return weight


if __name__ == "__main__":
    science_data_path = "../data/science/paths.pkl"
    money_data_path = "../data/money/paths.pkl"
    feature = EdgeProvenanceFeature()
    run_edge_feature_on_original(edge_feature=feature,
                                 data_path=science_data_path,
                                 out=f"../data/science/features/{FEATURE_NAME}.pkl")
    run_edge_feature_on_original(edge_feature=feature,
                                 data_path=money_data_path,
                                 out=f"../data/money/features/{FEATURE_NAME}.pkl")
    
    fixed_endpoints_science_data_path = "../data/fixed_endpoints/science_paths_fixed_endpoints.pkl"
    fixed_endpoints_money_data_path = "../data/fixed_endpoints/money_paths_fixed_endpoints.pkl"
    run_edge_feature_on_fixed_endpoints(edge_feature=feature,
                                        data_path=fixed_endpoints_money_data_path,
                                        out=f"../data/fixed_endpoints/money_features/{FEATURE_NAME}.pkl")
    run_edge_feature_on_fixed_endpoints(edge_feature=feature,
                                        data_path=fixed_endpoints_science_data_path,
                                        out=f"../data/fixed_endpoints/science_features/{FEATURE_NAME}.pkl")
