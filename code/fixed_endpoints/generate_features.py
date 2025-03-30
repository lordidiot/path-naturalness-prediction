import sys
import pickle

from edge_features import get_edge_features
from vertex_features import get_vertex_features

from .utils import load_pickle

DATA_FOLDER = "../../data/fixed_endpoints/"

def get_vertexes_and_edges(path_data_filepath: str):
    data = load_pickle(path_data_filepath)
    vtxs = set()
    edges = set()
    
    for vtx_pair, paths in data.items():
        vtxs.add(vtx_pair[0])
        vtxs.add(vtx_pair[1])
        
        for path in paths:
            for i, edge in enumerate(path.edge_list):
                edges.add(edge)
        
    return vtxs, edges

def save_features(features_folder, vtx_deg, vtx_emb, vtx_freq, edge_sim):
    with open(features_folder + "vertex_degree.pkl", "wb") as file:
        pickle.dump(vtx_deg, file)
        
    with open(features_folder + "vertex_embedding.pkl", "wb") as file:
        pickle.dump(vtx_emb, file)
        
    with open(features_folder + "vertex_frequency.pkl", "wb") as file:
        pickle.dump(vtx_freq, file)
    
    with open(features_folder + "edge_ends_similarity.pkl", "wb") as file:
        pickle.dump(edge_sim, file)

    with open(features_folder + "edge_direction.pkl", "wb") as file:
        pickle.dump(edge_sim, file)
    

def main(args):
    if len(args) == 0:
        dataset = "science"
    else:
        dataset = args[0]
    path_data_filepath = DATA_FOLDER + f"{dataset}_paths_fixed_endpoints.pkl"
    features_folder = DATA_FOLDER + f"{dataset}_features/"
    vertexs, edges = get_vertexes_and_edges(path_data_filepath)
    vtx_deg, vtx_emb, vtx_freq = get_vertex_features(vertexs)
    edge_sim, edge_dir = get_edge_features(edges, vtx_emb)
    save_features(features_folder, vtx_deg, vtx_emb, vtx_freq, edge_sim, edge_dir)

if __name__ == "__main__":
    main(sys.argv)
