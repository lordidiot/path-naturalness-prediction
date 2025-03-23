import requests
from collections import defaultdict
import numpy as np
import pickle

class Edge:
    start_id: str
    end_id: str
    short: str
    text: str

    def __init__(self, start_id, end_id, short, text):
        self.start_id = start_id
        self.end_id = end_id
        self.short = short
        self.text = text

class Node:
    # NOTE: id might be different from prefix + label, should keep both fields
    id: str
    label: str
    out_edge_list: list[Edge]

    def __init__(self, id, label, out_edge_list):
        self.id = id
        self.label = label
        self.out_edge_list = out_edge_list
    
    def __str__(self):
        return self.label
    
    def __repr__(self):
        return self.label
    
    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        # some nodes have same label but different id. should be considered equal as well.
        # TODO: labels that belong to the same stem should be considered equal
        return self.id == other.id or self.label == other.label

class Path:
    start: Node
    end: Node
    length: int

    node_list: list[Node]
    edge_list: list[Edge]

    def __init__(self, start, end, length, node_list, edge_list):
        self.start = start
        self.end = end
        self.length = length
        self.node_list = node_list
        self.edge_list = edge_list

    def extend(self, edge: Edge, next_node: Node):
        assert self.end.id == edge.start_id and edge.end_id == next_node.id

        new_node_list = self.node_list.copy()
        new_node_list.append(next_node)
        new_edge_list = self.edge_list.copy()
        new_edge_list.append(edge)

        return Path(self.start, next_node, self.length + 1, new_node_list, new_edge_list)
    
    def short(self):
        ret = f'{self.start.label}'
        for n, e in zip(self.node_list[1:], self.edge_list):
            ret += f' <--{e.short}--> {n.label}'
        return ret
    
    def text(self):
        ret = ''
        for e in self.edge_list:
            ret += f'{e.long}. '
        return ret
        
    def __str__(self):
        return self.short()
    
    def __repr__(self):
        return self.short()


URL = "https://api.conceptnet.io"
# ID_PREFIX = "/e/en/"
# CONCEPT_NET_URL_PREFIX = URL + ID_PREFIX

memoised_nodes = {}

def get_node(id):
    if id in memoised_nodes:
        return memoised_nodes[id]
    response = requests.get(URL + id)
    if response.status_code != 200:
        print(f'Error: Cannot get node {id}, error code {response.status_code}')
        return None
    response_json = response.json()
    
    edges = response_json['edges']
    if len(edges) == 0:
        print(f'Warning: Node {id} has no edges')
        return None
    label = edges[0]['start']['label'] if edges[0]['start']['@id'] == id else edges[0]['end']['label']

    def filter_edges(dict):
        # TODO: add more filters if necessary, e.g. only easy / commonly used words
        return dict['start']['@id'] == id and \
               dict['end'].get('language', None) == 'en' and \
               len(dict['end']['label'].split(' ')) <= 2 and \
               len(dict['end']['label']) > 2

    edges = list(filter(filter_edges, edges))
    
    def edge_dict_to_object(dict):
        start_id = dict['start']['@id']
        end_id = dict['end']['@id']
        short = dict['rel']['label']
        long = dict['surfaceText']

        return Edge(start_id, end_id, short, long)

    out_edge_list = list(map(edge_dict_to_object, edges))
    
    node = Node(id, label, out_edge_list)
    memoised_nodes[id] = node
    return node


def random_walk(node, samples: set, num_to_sample):
    # from a node, gather nearby nodes by random walk
    # stop when num_to_sample nodes are gathered
    if len(samples) >= num_to_sample:
        return
    
    if node in samples:
        return
    
    print(f'Got new node: {node.label}')
    samples.add(node)

    n = min(5, len(node.out_edge_list))
    edges = np.random.choice(node.out_edge_list, n, replace=False)
    for edge in edges:
        next_node = get_node(edge.end_id)
        random_walk(next_node, samples, num_to_sample)


def generate_paths_bfs(node, steps_min, steps_max):
    # generate all non-cyclic paths from a node, length in [steps_min, steps_max]
    queue = [Path(node, node, 0, [node], [])]
    ret = []
    while queue:
        path = queue.pop(0)
        if path.length >= steps_min:
            ret.append(path)
        if path.length < steps_max:
            # continue BFS
            current_node_ids = list(map(lambda x: x.id, path.node_list))
            for edge in path.end.out_edge_list:
                # avoid cyclic
                if edge.end_id not in current_node_ids:
                    next_node = get_node(edge.end_id)
                    queue.append(path.extend(edge, next_node))
    return ret


def aggregate_by_same_end_node(paths: list[Path]):
    # for a list of paths with same start node, aggregate by end node
    # return dict sorted by desc aggregate count of each end node
    ret = defaultdict(list)
    for path in paths:
        ret[path.end.id].append(path)
    return sorted(ret.items(), key=lambda item: len(item[1]), reverse=True)


BFS_STEP_MIN = 2
BFS_STEP_MAX = 5
EACH_PAIR_PATHS_COUNT = 10 # also the threshold for being a partner

def find_partners_from_initial_nodes(initial_nodes: list[Node], num_partners_per_node):
    # for each initial node, use BFS to find its "partners"
    # partner of a start node, means an end node that has >= {EACH_PAIR_PATHS_COUNT} different paths from it
    # store as dict, key is (start_node_id, partner_id), value is list of {EACH_PAIR_PATHS_COUNT} paths between them
    ret = defaultdict(list)
    count = 0
    for node in initial_nodes:
        print(f'Generating paths from node {node.id}')
        paths = generate_paths_bfs(node, BFS_STEP_MIN, BFS_STEP_MAX)
        paths_agg_by_end_node = aggregate_by_same_end_node(paths)
        if len(paths_agg_by_end_node) == 0:
            print(f'Warning: node {node.id} is leaf, no partners')
            continue
        
        # this would be in descending order of aggregate count
        for i, item in enumerate(paths_agg_by_end_node):
            if i >= num_partners_per_node:
                break
            partner_node_id, paths = item
            if len(paths) < EACH_PAIR_PATHS_COUNT:
                # this did not qualify as partner
                # and since descending order, no more partners for this start node
                print(f'Warning: node {node.id} only has {i} partners')
                break
            # partner found!
            count += 1
            key = (node.id, partner_node_id)
            for path in np.random.choice(paths, EACH_PAIR_PATHS_COUNT, replace=False):
                ret[key].append(path)
        print(f'Found {count} partner pairs so far')
    return ret


SCIENCE_INITIAL_NUM_NODES = 1000
SCIENCE_EACH_NODE_NUM_PARTNERS = 10
def generate_from_science():
    science = get_node('/c/en/science')
    initial_nodes = set()
   
    random_walk(science, initial_nodes, SCIENCE_INITIAL_NUM_NODES)
    
    initial_nodes = list(initial_nodes)
    print(f'Initial nodes from science: {initial_nodes}')        
    
    partners_dict = find_partners_from_initial_nodes(initial_nodes, SCIENCE_EACH_NODE_NUM_PARTNERS)
    return partners_dict


MONEY_INITIAL_NUM_NODES = 1000
MONEY_EACH_NODE_NUM_PARTNERS = 1
def generate_from_money():
    money = get_node('/c/en/money')
    initial_nodes = set()
   
    random_walk(money, initial_nodes, MONEY_INITIAL_NUM_NODES)
    
    initial_nodes = list(initial_nodes)
    print(f'Initial nodes from money: {initial_nodes}')
    
    partners_dict = find_partners_from_initial_nodes(initial_nodes, MONEY_EACH_NODE_NUM_PARTNERS)
    return partners_dict


ALLOWED_POS = ['n', 'v', 'j']
OPEN_DOMAIN_INITIAL_NUM_NODES = 10000
OPEN_DOMAIN_EACH_NODE_NUM_PARTNERS = 1
# https://www.wordfrequency.info/samples/lemmas_60k_words.txt
def generate_from_open_domain():
    lines = []
    with open('data/fixed-endpoints/words.txt', 'r') as f:
        lines = f.readlines()[9:]
        f.close()
    lines = [line.strip().split('\t') for line in lines]
    
    initial_nodes = []
    prev_lemma = ''
    for line in lines:
        lemma = line[1]
        pos = line[2]
        if lemma == prev_lemma or pos not in ALLOWED_POS:
            continue
        node = get_node(f'/c/en/{lemma}')
        if node is not None:
            initial_nodes.append(node)
        if len(initial_nodes) >= OPEN_DOMAIN_INITIAL_NUM_NODES:
            break
        prev_lemma = lemma
    
    partners_dict = find_partners_from_initial_nodes(initial_nodes, OPEN_DOMAIN_EACH_NODE_NUM_PARTNERS)
    return partners_dict

    
if __name__ == '__main__':
    science_data = generate_from_science()
    with open('data/fixed-endpoints/science_paths_fixed_endpoints.pkl', 'wb') as f:
        pickle.dump(science_data, f)
        f.close()
    
    money_data = generate_from_money()
    with open('data/fixed-endpoints/money_paths_fixed_endpoints.pkl', 'wb') as f:
        pickle.dump(money_data, f)
        f.close()
    
    open_domain_data = generate_from_open_domain()
    with open('data/fixed-endpoints/open_domain_paths_fixed_endpoints.pkl', 'wb') as f:
        pickle.dump(open_domain_data, f)
        f.close()
