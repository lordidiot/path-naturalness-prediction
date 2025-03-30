import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_edge_features(edges, vtx_emb):
    edge_sim = dict()
    
    for edge in edges:
        vtx1_emb = vtx_emb[edge.lhs_name]
        vtx2_emb = vtx_emb[edge.rhs_name]
        vtx_pair = tuple(sorted((edge.lhs_name, edge.rhs_name)))
        
        if vtx_pair not in edge_sim:
            edge_sim[vtx_pair] = cosine_similarity(vtx1_emb, vtx2_emb)

    return edge_sim
