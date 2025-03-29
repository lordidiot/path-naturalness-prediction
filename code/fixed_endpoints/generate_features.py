import sys

from .utils import load_pickle
from vertex_features import get_vertex_features

def get_vertexes(path_data_filepath: str):
    data = load_pickle(path_data_filepath)
    vtxs = set()
    for nodes in data:
        vtxs.add(nodes[0])
        vtxs.add(nodes[1])
    return vtxs
    
def main(args):
    if len(args) == 0:
        path_data_filepath = "../../data/fixed_endpoints/science_paths_fixed_endpoints.pkl"
    else:
        path_data_filepath = args[0]
    vertexs = get_vertexes(path_data_filepath)
    get_vertex_features(vertexs)

if __name__ == "__main__":
    main(sys.argv)
