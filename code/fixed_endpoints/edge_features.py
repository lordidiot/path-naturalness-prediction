import numpy as np
import requests

RELATIONS = [
    "RelatedTo", "FormOf", "IsA", "PartOf", "HasA", "UsedFor", "CapableOf",
    "AtLocation", "Causes", "HasSubevent", "HasFirstSubevent", "HasLastSubevent",
    "HasPrerequisite", "HasProperty", "MotivatedByGoal", "ObstructedBy", "Desires",
    "CreatedBy", "Synonym", "Antonym", "DistinctFrom", "DerivedFrom", "SymbolOf",
    "DefinedAs", "MannerOf", "LocatedNear", "HasContext", "SimilarTo", "EtymologicallyRelatedTo",
    "EtymologicallyDerivedFrom", "CausesDesire", "MadeOf", "ReceivesAction", "ExternalURL"
]

SOURCES = [
    "/s/resource/wordnet", "/s/resource/dbpedia", "/s/resource/verbosity", 
    "/s/resource/wiktionary", "/s/resource/opencyc", "/s/contributor/omcs"
]

# wordnet, dbpedia, verbosity, wiktionary, opencyc, omcs
SOURCES_MOST_COMMON_WEIGHT = [2.0, 1.0, 0.102, 1.0, 1.0, 1.0]

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

def edge_provenance(edge):
    node1 = edge.lhs_name
    node2 = edge.rhs_name
    rel = edge.short.split("--")[1]
    obj = requests.get(f"http://api.conceptnet.io/query?node=/c/en/{node1}&other=/c/en/{node2}&/rel=/r{rel}").json()
    
    sources = obj["edges"]["sources"]
    idxes = set()
    for source in sources:
        for i, s in SOURCES:
            if source["contributor"].startswith(s):
                idxes.add(i)
    
    edge_prov = [0] * len(SOURCES)
    for i in idxes:
        edge_prov[i] = SOURCES_MOST_COMMON_WEIGHT[i]
    
    return edge_prov
    
def get_edge_features(edges, vtx_emb):
    relation_one_hot = {
        relation: np.eye(len(RELATIONS), dtype=int)[i] for i, relation in enumerate(RELATIONS)
    }
    
    edge_sim = dict()
    edge_dir = dict()
    edge_rel = dict()
    edge_prov = dict()
    
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
        
        # edge provenance
        if edge not in edge_prov:
            edge_prov[edge] = edge_provenance(edge)
        
    return edge_sim, edge_dir, edge_rel, edge_prov
