import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import requests

# >>> paths['4444']['forward']['short']
# 'Paper <--Antonym--> Card <--RelatedTo--> Hand <--RelatedTo--> Food '
# path = 'Paper <--Antonym--> Card <--RelatedTo--> Hand <--RelatedTo--> Food '
# pprint(path.rstrip().split(' '))

# >>> e_dir['4444f']
# [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

# <rant>The paper says they use the 840B version, but the data they
#       uploaded corresponds to the 42B version?!?!?!?! </rant>
GLOVE_PATH = 'glove.42B.300d.txt' # https://nlp.stanford.edu/data/glove.42B.300d.zip
_glove_cached = None
def get_glove_embeddings():
    global _glove_cached
    if _glove_cached:
        return _glove_cached
    if not os.path.exists(GLOVE_PATH):
        print(f"glove embeddings not found at path: {GLOVE_PATH}", file=sys.stderr)
        sys.exit(1)
    _glove_cached = {}
    with open(GLOVE_PATH, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip().split(' ')
            _glove_cached[line[0]] = np.array(line[1:], dtype=np.float32)
    return _glove_cached

def edge_id(v1, v2, rel) -> tuple[str, str, str]:
    if '-' in rel:
        rel = rel.split('-')[2]
    return tuple(sorted((v1.lower(), v2.lower()))) + (rel,)

def enumerate_words(paths):
    words = set()
    for path in paths.values():
        words.update(i.lower() for i in path['forward']['short'].rstrip().split(' ')[::2])
    return words

def enumerate_edges(paths):
    edges = set()
    for path in paths.values():
        path_split = path['forward']['short'].rstrip().split(' ')
        for i in range(0, len(path_split)-1, 2):
            v1, edge, v2 = path_split[i:i+3]
            edges.add(edge_id(v1, v2, edge))
    return edges

def _cache_word_embeddings():
    with open('../../data/science/paths.pkl', 'rb') as f:
        paths = pickle.load(f)
    science_words = enumerate_words(paths)
    all_glove_embeddings = get_glove_embeddings()
    glove_embeddings = {}
    for word in science_words:
        glove_embeddings[word] = all_glove_embeddings[word]
        glove_embeddings[word.lower()] = all_glove_embeddings[word.lower()]
    with open('./glove_embeddings_cached.pkl', 'wb') as f:
        pickle.dump(glove_embeddings, f)

CONCEPTNET_API_URL = "https://api.conceptnet.io"
def get_conceptnet_data(words: list[str]):
    print("Getting ConceptNet data for words")
    if os.path.exists('./conceptnet_data_cached.pkl'):
        with open('./conceptnet_data_cached.pkl', 'rb') as f:
            return pickle.load(f)
    conceptnet_data = {}
    for word in tqdm(words):
        word = word.lower()
        obj = requests.get(f"{CONCEPTNET_API_URL}/c/en/{word}").json()
        conceptnet_data[word] = obj
    with open('./conceptnet_data_cached.pkl', 'wb') as f:
        return pickle.dump(conceptnet_data, f)
    return conceptnet_data

def get_conceptnet_edge_data(edges: list[tuple[str, str, str]]):
    print(f"Getting ConceptNet data for {len(edges)} edges")
    if os.path.exists('./conceptnet_edge_data_cached.pkl'):
        with open('./conceptnet_edge_data_cached.pkl', 'rb') as f:
            return pickle.load(f)
    edge_data = {}
    for edge in tqdm(edges):
        a, b, rel = edge
        data = requests.get(f"{CONCEPTNET_API_URL}/query?node=/c/en/{a}&other=/c/en/{b}&rel=/r/{rel}").json()
        if not len(data['edges']):
            print("Something wong:", edge)
        else:
            edge_data[edge] = data['edges'][0]
    with open('./conceptnet_edge_data_cached.pkl', 'wb') as f:
        return pickle.dump(edge_data, f)
    return edge_data

def get_sources(edge):
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

def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <paths.pkl> <output_dir>', file=sys.stderr)
        return
    
    paths_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    with open(paths_path, 'rb') as f:
        paths = pickle.load(f)
    if not output_dir.exists():
        output_dir.mkdir()
    words = enumerate_words(paths)
    edges = enumerate_edges(paths)
    conceptnet_data = get_conceptnet_data(words)
    conceptnet_edge_data = get_conceptnet_edge_data(edges)

    # v_enc_dim300 (Glove embeddings)
    output_file = output_dir / 'v_enc_dim300.pkl'
    if not output_file.exists():
        print(f"Saving glove embeddings to {str(output_file)}...")
        if os.path.exists('./glove_embeddings_cached.pkl'):
            with open('./glove_embeddings_cached.pkl', 'rb') as f:
                glove_embeddings = pickle.load(f)
        else:
            glove_embeddings = get_glove_embeddings()
        v_enc_glove = {}
        for path_id in paths:
            words = paths[path_id]['forward']['short'].rstrip().split(' ')[::2]
            forward_encoding = list(glove_embeddings[word.lower()] for word in words)
            v_enc_glove[path_id+'f'] = forward_encoding
            v_enc_glove[path_id+'r'] = forward_encoding[::-1]
        with open(output_file, 'wb') as f:
            pickle.dump(v_enc_glove, f)
    else:
        print(f"Glove embeddings {str(output_file)} already exist. Skipping.")

    # v_freq_freq
    # v_enc_dim300
    # v_freq_freq
    # v_deg
    # v_sense

    # e_vertexsim (Edge ends similarity)
    # Paper claims to use similarity, but actually uses distance
    output_file = output_dir / 'e_vertexsim.pkl'
    if not output_file.exists():
        print(f"Saving edge ends similarity to {str(output_file)}...")
        with open(output_dir / 'v_enc_dim300.pkl', 'rb') as f:
            v_enc_glove = pickle.load(f)
        e_vertexsim = {}
        for path_id in v_enc_glove:
            v = v_enc_glove[path_id]
            esims = list(
                1 - np.dot(v[i], v[i+1]) / (np.linalg.norm(v[i]) * np.linalg.norm(v[i+1])) \
                for i in range(len(v)-1)
            )
            e_vertexsim[path_id] = esims
        with open(output_file, 'wb') as f:
            pickle.dump(e_vertexsim, f)
    else:
        print(f"Edge ends similarity {str(output_file)} already exist. Skipping.")

    # e_dir
    # e_rel

    # e_weightsource
    output_file = output_dir / 'e_weightsource.pkl'
    if not output_file.exists():
        print(f"Saving edge provenance to {str(output_file)}...")
        common_weights = [0.5, 1.0, 1.0, 0, 1.0, 2.0]
        all_sources = ['/s/resource/dbpedia', '/s/contributor/omcs', '/s/resource/opencyc/2012', '/s/resource/verbosity', '/s/resource/wiktionary', '/s/resource/wordnet/rdf/3.1']
        e_weightsource = {}
        for path_id in paths:
            weightsource = []
            path = paths[path_id]['forward']['short'].rstrip().split(' ')
            for i in range(0, len(path)-1, 2):
                v1, edge, v2 = path[i:i+3]
                _edge_id = edge_id(v1, v2, edge)
                if _edge_id not in conceptnet_edge_data:
                    weightsource.append([0, 0, 0, 0, 0, 0])
                    continue
                weight = [0, 0, 0, 0, 0, 0]
                edge_data = conceptnet_edge_data[_edge_id]
                sources = get_sources(edge_data)
                if len(sources) == 1:
                    weight[all_sources.index(sources[0])] = edge_data['weight']
                elif set(sources) == set(['/s/resource/verbosity', '/s/resource/wiktionary']):
                    weight[4] = common_weights[4]
                    weight[3] = edge_data['weight'] - common_weights[4]
                else:
                    print(f"Can't handle combination: {sources}")
                    return
                weightsource.append(weight)
            e_weightsource[path_id+'f'] = weightsource
            e_weightsource[path_id+'r'] = weightsource[::-1]
        with open(output_file, 'wb') as f:
            pickle.dump(e_weightsource, f)
    else:
        print(f"Edge provenance {str(output_file)} already exist. Skipping.")



    # e_srank_rel
    # e_trank_rel
    # e_sense
    

if __name__ == '__main__':
    main()

