import numpy as np
import pickle

GLOVE_PATH = "../data/glove.42B.300d.txt"

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
        
def get_vertex_features(vtxs: set[str], features_folder: str):
    vtx_emb, vtx_freq = get_glove_features(vtxs)
    
    with open(features_folder + 'vertex_embedding.pkl', 'wb') as file:
        pickle.dump(vtx_emb, file)
        
    with open(features_folder + 'vertex_frequency.pkl', 'wb') as file:
        pickle.dump(vtx_freq, file)
    
    