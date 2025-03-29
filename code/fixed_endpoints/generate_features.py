import sys

from .utils import load_pickle
from vertex_features import get_vertex_features

DATA_FOLDER = "../../data/fixed_endpoints/"

def get_vertexes(path_data_filepath: str):
    data = load_pickle(path_data_filepath)
    vtxs = set()
    for nodes in data:
        vtxs.add(nodes[0])
        vtxs.add(nodes[1])
    return vtxs
    
def main(args):
    if len(args) == 0:
        dataset = "science"
    else:
        dataset = args[0]
    path_data_filepath = DATA_FOLDER + f"{dataset}_paths_fixed_endpoints.pkl"
    features_folder = DATA_FOLDER + f"{dataset}_features/"
    vertexs = get_vertexes(path_data_filepath)
    get_vertex_features(vertexs, features_folder)

if __name__ == "__main__":
    main(sys.argv)
