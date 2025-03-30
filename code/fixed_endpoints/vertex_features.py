import numpy as np
import pickle
import requests

GLOVE_PATH = "../data/glove.42B.300d.txt"
LIMIT = 1000

def get_glove_features(vtxs: set[str]):
    vtx_emb = {}
    vtx_freq = {}
    
    with open(GLOVE_PATH, 'r') as file:
        for i, row in enumerate(file):
            row = row.rstrip().split(' ')
            word = row[0]
            if word in vtxs:
                vtx_freq[word] = 1 / (i + 1)
            vtx_emb[word] = np.array(row[1:], dtype=np.float32)
    
    return vtx_emb, vtx_freq

# number of neighbours with no duplicates (even if there are multiple relations)
def get_vertex_degree(vtxs: set[str]):
    vtx_deg = {}
    
    for vtx in vtxs:
        offset = 0
        neighbours = set()
        isEnd = False
        while not isEnd:
            obj = requests.get(f"http://api.conceptnet.io/c/en/{vtx}?offset={offset}&limit={LIMIT}").json()
            edges = obj["edges"]
            if len(edges) < LIMIT:
                isEnd = True
            for edge in edges:
                start = edge["start"]["label"]
                end = edge["end"]["label"]
                if vtx == start:
                    neighbour = end
                elif vtx == end:
                    neighbour = start
                else:
                    print("ERROR: Vertex is not one of nodes")
                
                neighbours.add(neighbour)
            offset += LIMIT
        
        vtx_deg[vtx] = len(neighbours)
        
    return vtx_deg
        
        
        
def get_vertex_features(vtxs: set[str], features_folder: str):
    vtx_deg = get_vertex_degree(vtxs)
    vtx_emb, vtx_freq = get_glove_features(vtxs)
    
    with open(features_folder + "vertex_degree.pkl", "wb") as file:
        pickle.dump(vtx_deg, file)
        
    with open(features_folder + "vertex_embedding.pkl", "wb") as file:
        pickle.dump(vtx_emb, file)
        
    with open(features_folder + "vertex_frequency.pkl", "wb") as file:
        pickle.dump(vtx_freq, file)
    
    