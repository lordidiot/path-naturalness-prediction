import numpy as np

RELATIONS = [
    "RelatedTo", "Synonym", "Antonym", "DistinctFrom", "LocatedNear", "SimilarTo",
    "EtymologicallyRelatedTo", "FormOf", "IsA", "PartOf", "HasA", "UsedFor", "CapableOf",
    "AtLocation", "Causes", "HasSubevent", "HasFirstSubevent", "HasLastSubevent",
    "HasPrerequisite", "HasProperty", "MotivatedByGoal", "ObstructedBy", "Desires",
    "CreatedBy", "DerivedFrom", "SymbolOf", "DefinedAs", "MannerOf", "HasContext",
    "EtymologicallyDerivedFrom", "CausesDesire", "MadeOf", "ReceivesAction", "ExternalURL",
]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def edge_direction(relation: str):
    # (forward, backward, bidirectional)
    if relation[-1] == '>':
        if relation[0] == '<':
            return [0,0,1]
        else:
            return [1,0,0]
    elif relation[0] == '<':
        return [0,1,0]
    else:
        print(f"Invalid relation: {relation}")
        return [0,0,0] 

def get_edge_features(edges, vtx_emb):
    relation_one_hot = {
        relation: np.eye(len(RELATIONS), dtype=int)[i] for i, relation in enumerate(RELATIONS)
    }
    
    edge_sim = dict()
    edge_dir = dict()
    edge_rel = dict()
    
    for edge in edges:
        # edge direction
        if edge not in edge_dir:
            edge_dir[edge] = edge_direction(edge.short)
            
        # edge ends similarity
        vtx1_emb = vtx_emb[edge.lhs_name]
        vtx2_emb = vtx_emb[edge.rhs_name]
        
        if edge not in edge_sim:
            edge_sim[edge] = cosine_similarity(vtx1_emb, vtx2_emb)
        
        # edge relation
        if edge not in edge_rel:
            edge_rel[edge] = relation_one_hot[edge.short.strip("<>--")]
        
        
    return edge_sim, edge_dir, edge_rel
