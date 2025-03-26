class Edge:
    # edges have a direction: -->, <--. or <-->
    # an edge is well defined by lhs_name, rhs_name and short
    lhs_name: str
    rhs_name: str
    short: str # arrow representation including direction information
    text: str
    short_reverse: str
    text_reverse: str

    def __init__(self, lhs_name, rhs_name, short, text):
        self.lhs_name = lhs_name
        self.rhs_name = rhs_name
        self.short = short
        self.text = text

        left = short[0] == '<'
        right = short[-1] == '>'
        self.short_reverse = f"{'<' if right else ''}--{self.short.strip('<>-')}--{'>' if left else ''}"

        if text is None:
            self.text_reverse = None
        else:
            text_middle = text.split(']]')[1].split('[[')[0]
            self.text_reverse = f"[[{self.rhs_name}]]{text_middle}[[{self.lhs_name}]]"


class Node:
    # represents a vertex in the graph
    # note these naming conventions:
    ### id means the @id field on conceptnet api, in the format of /c/en/{name}
    ### name means the word or phrase after prefix /c/en/
    ### label means the label field on conceptnet api (may be different from name)
    name: str
    edge_list: list[Edge] # all incident edges including all 3 directions

    def __init__(self, name, edge_list):
        self.name = name
        self.edge_list = edge_list
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return self.name == other.name

class Path:
    id: str # should prefix with 'cs4248/'
    # Path is created with a certain direction, relevant fields follow this direction
    # The reverse path representations can be obtained with reverse=True in the instance methods
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
        # creates a new path by extending the current path with an edge to the next node
        assert edge.lhs_name == self.end.name and edge.rhs_name == next_node.name

        new_node_list = self.node_list.copy()
        new_node_list.append(next_node)
        new_edge_list = self.edge_list.copy()
        new_edge_list.append(edge)

        return Path(self.start, next_node, self.length + 1, new_node_list, new_edge_list)
    
    def short(self, reverse=False):
        ret = f'{self.start.name}'
        for n, e in zip(self.node_list[1:], self.edge_list):
            if reverse:
                ret = f'{n.name} {e.short_reverse} {ret}'
            else:
                ret = f'{ret} {e.short} {n.name}'
        return ret
    
    def text(self, reverse=False):
        ret = ''
        for e in self.edge_list:
            if reverse:
                ret = f'{e.text_reverse}. {ret}'
            else:
                ret = f'{ret} {e.text}.'
        return ret
        
    def __str__(self):
        return self.short()
    
    def __repr__(self):
        return self.short()
