import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

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

def enumerate_words(paths):
    words = set()
    for path in paths.values():
        words.update(i.lower() for i in path['forward']['short'].rstrip().split(' ')[::2])
    return words

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
    # e_srank_rel
    # e_trank_rel
    # e_sense
    

if __name__ == '__main__':
    main()

