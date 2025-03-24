import requests
from collections import defaultdict
import random
import pickle
from .types import Node, Edge, Path

URL = "https://api.conceptnet.io"
ID_PREFIX = "/c/en/"
CONCEPT_NET_URL_PREFIX = URL + ID_PREFIX

# https://github.com/commonsense/conceptnet5/wiki/relations
BIDIRECTIONAL_REL_ID = ['/r/RelatedTo', '/r/Synonym', '/r/Antonym', '/r/DistinctFrom', '/r/LocatedNear', '/r/SimilarTo', '/r/EtymologicallyRelatedTo']

common_noun_lemmas = []
with open('fixed_endpoints/common_noun_lemmas.txt', 'r') as f:
    for line in f.readlines()[:1000]:
        common_noun_lemmas.append(line.strip())
    f.close()

memoised_nodes = {}

def get_node(name):
    if name in memoised_nodes:
        return memoised_nodes[name]
    
    def get_edges(name, offset, limit):
        response = requests.get(f'{CONCEPT_NET_URL_PREFIX}{name}?offset={offset}&limit={limit}')
        response_json = response.json()
        return response_json['edges']
    
    # get all edges
    offset = 0
    limit = 1000
    edges = []
    while True:
        new_edges = get_edges(name, offset, limit)
        if not new_edges or len(new_edges) == 0:
            break
        edges.extend(new_edges)
        offset += limit
    
    if len(edges) == 0:
        print(f'Warning: Node {name} has no edges')
        return None

    def filter_edges(dict):
        start_name = dict['start']['@id'].strip('/').split('/')[2]
        other = dict['end'] if name == start_name else dict['start']
        other_name = other['@id'].strip('/').split('/')[2]

        return other.get('language', None) == 'en' and \
               other_name in common_noun_lemmas

    edges = list(filter(filter_edges, edges))
    
    def edge_dict_to_object(dict):
        start_name = dict['start']['@id'].strip('/').split('/')[2]
        end_name = dict['end']['@id'].strip('/').split('/')[2]
        lhs_name = name
        rhs_name = end_name if name == start_name else start_name
        text = dict['surfaceText']
        short = ''

        if dict['rel']['@id'] in BIDIRECTIONAL_REL_ID:
            short = f'<--{dict["rel"]["label"]}-->'
        elif start_name == name:
            short = f'--{dict["rel"]["label"]}-->'
        else:
            short = f'<--{dict["rel"]["label"]}--'

        return Edge(lhs_name, rhs_name, short, text)

    edge_list = list(map(edge_dict_to_object, edges))
    
    node = Node(name, edge_list)
    memoised_nodes[name] = node
    return node


def random_walk(node: Node, samples: set, num_to_sample: int, words_to_avoid: list[str] = None):
    # from a node, gather nearby nodes by random walk
    # stop when num_to_sample nodes are gathered
    if len(samples) >= num_to_sample:
        return
    
    if node in samples:
        return
    
    print(f'Got new node: {node.name}')
    samples.add(node)

    n = min(5, len(node.edge_list)) # needed to avoid leaf causing early termination
    edges = node.edge_list.copy()
    random.shuffle(edges)
    edges = edges[:n]
    for edge in edges:
        if words_to_avoid and edge.rhs_name in words_to_avoid:
            continue
        next_node = get_node(edge.rhs_name)
        random_walk(next_node, samples, num_to_sample)


def generate_paths_in_subgraph_bfs(node: Node, subgraph_node_names: list[str], steps_min: int, steps_max: int) -> list[Path]:
    # in the subgraph spanned by {nodes_in_subgraph},
    # generate all non-cyclic paths from a node, length in [steps_min, steps_max]
    assert node.name in subgraph_node_names
    queue = [Path(node, node, 0, [node], [])]
    ret = []
    while queue:
        path = queue.pop(0)
        if path.length >= steps_min:
            ret.append(path)
        if path.length < steps_max:
            # continue BFS
            current_node_names = list(map(lambda x: x.name, path.node_list))
            for edge in path.end.edge_list:
                if edge.rhs_name not in subgraph_node_names:
                    continue
                if edge.rhs_name in current_node_names:
                    # avoid cyclic
                    continue
                next_node = get_node(edge.rhs_name) # should be memoised
                queue.append(path.extend(edge, next_node))
    return ret


def aggregate_by_same_end_node(paths: list[Path]) -> list[tuple[str, list[Path]]]:
    # for a list of paths with same start node, aggregate by end node
    # return dict sorted by desc aggregate count of each end node
    ret = defaultdict(list)
    for path in paths:
        ret[path.end.name].append(path)
    return sorted(ret.items(), key=lambda item: len(item[1]), reverse=True)


BFS_STEP_MIN = 1
BFS_STEP_MAX = 4
EACH_PAIR_PATHS_COUNT = 10 # also the threshold for being a partner

count = 0
def find_partners_from_initial_nodes(initial_nodes: list[Node], num_partners_per_node):
    # for each initial node, use BFS to find its "partners"
    # partner of a start node, means an end node that has >= {EACH_PAIR_PATHS_COUNT} different paths from it
    # store as dict, key is (start_node_name, partner_name), value is list of {EACH_PAIR_PATHS_COUNT} paths between them
    global count
    initial_nodes_names = list(map(lambda x: x.name, initial_nodes))
    ret = defaultdict(list)
    for node in initial_nodes:
        print(f'Generating paths from node {node.name}')
        paths = generate_paths_in_subgraph_bfs(node, initial_nodes_names, BFS_STEP_MIN, BFS_STEP_MAX)
        paths_agg_by_end_node = aggregate_by_same_end_node(paths)
        if len(paths_agg_by_end_node) == 0:
            print(f'Warning: node {node.name} is leaf, no partners')
            continue
        
        # this is the list of potential partners
        paths_agg_by_end_node = list(filter(lambda item: len(item[1]) >= EACH_PAIR_PATHS_COUNT, paths_agg_by_end_node))
        if len(paths_agg_by_end_node) < num_partners_per_node:
            print(f'Warning: node {node.name} only has {len(paths_agg_by_end_node)} partners')
        
        # randomly sample partners
        # NOTE: distribution could be biased towards far-away partners
        # (because being far from the start node means more paths and more likely to be partner)
        # this means the paths generated could be mostly long and unnatural
        #
        # the following is an attempt to mitigate the issue
        # for each partner, attempt to sample shorter paths first before longer paths
        n = min(num_partners_per_node, len(paths_agg_by_end_node))
        random.shuffle(paths_agg_by_end_node)
        paths_agg_by_end_node = paths_agg_by_end_node[:n]
        for partner_node_name, paths in paths_agg_by_end_node:
            key = (node.name, partner_node_name)
            short_threshold = (BFS_STEP_MIN + BFS_STEP_MAX) // 2 # "short" is defined as length <= short_threshold
            short_paths = list(filter(lambda x: x.length <= short_threshold, paths))
            n = min(EACH_PAIR_PATHS_COUNT // 2, len(short_paths)) # sample half from short paths, if available
            random.shuffle(short_paths)
            short_paths = short_paths[:n]
            for path in short_paths:
                path.id = f'cs4248/{count}'
                count += 1
                ret[key].append(path)
            long_paths = list(filter(lambda x: x.length > short_threshold, paths))
            n = EACH_PAIR_PATHS_COUNT - len(short_paths)
            random.shuffle(long_paths)
            long_paths = long_paths[:n]
            for path in long_paths:
                path.id = f'cs4248/{count}'
                count += 1
                ret[key].append(path)
        print(f'Found {count} paths so far')
    return ret


SCIENCE_INITIAL_NUM_NODES = 300
SCIENCE_EACH_NODE_NUM_PARTNERS = 10
science_initial_nodes = set()
def generate_from_science():
    global science_initial_nodes
    science = get_node('science')

    # Attempt to use same 100 initial nodes as the baseline
    # Did not work because 0 partner were found in the subgraph
    # with open('fixed_endpoints/science_initial_words.txt', 'r') as f:
    #     for line in f.readlines():
    #         science_initial_nodes.add(get_node(line.strip().lower()))
    #     f.close()
   
    random_walk(science, science_initial_nodes, SCIENCE_INITIAL_NUM_NODES)
    
    initial_nodes = list(science_initial_nodes)
    print(f'Initial nodes from science: {initial_nodes}')        
    
    partners_dict = find_partners_from_initial_nodes(initial_nodes, SCIENCE_EACH_NODE_NUM_PARTNERS)
    return partners_dict


MONEY_INITIAL_NUM_NODES = 100
MONEY_EACH_NODE_NUM_PARTNERS = 1
words_to_avoid = list(map(lambda x: x.name, science_initial_nodes))
def generate_from_money():
    money = get_node('money')
    initial_nodes = set()
   
    random_walk(money, initial_nodes, MONEY_INITIAL_NUM_NODES, words_to_avoid)
    
    initial_nodes = list(initial_nodes)
    print(f'Initial nodes from money: {initial_nodes}')
    
    partners_dict = find_partners_from_initial_nodes(initial_nodes, MONEY_EACH_NODE_NUM_PARTNERS)
    return partners_dict


if __name__ == '__main__':
    science_data = generate_from_science()
    with open('fixed_endpoints/science_paths_fixed_endpoints.pkl', 'wb') as f:
        pickle.dump(science_data, f)
        f.close()
    
    money_data = generate_from_money()
    with open('fixed_endpoints/money_paths_fixed_endpoints.pkl', 'wb') as f:
        pickle.dump(money_data, f)
        f.close()
