import pickle

from .utils import load_pickle

data = load_pickle("fixed_endpoints/science_paths_fixed_endpoints.pkl")

for count, j in enumerate(data.items()):
    for path in j[1]:
        for node in path.node_list:
            print(node)
    print(count)
