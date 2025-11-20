import networkx as nx
from typing import Iterable, Set, Tuple

def construct_closure(
        elements: Iterable, 
        relations: Set[Tuple]
    ) -> Tuple[nx.DiGraph, Set[Tuple]]:
    '''
    Constructs the transitive closure of a poset defined by its elements and relations.

    Args:
        elements (Iterable): A iterable of elements in the poset.
        relations (Set[Tuple]): A set of tuples representing the order relations between elements.

    Returns:
        Tuple[nx.DiGraph, Set[Tuple]]: A tuple containing the transitive closure graph and the set of edges in the closure.
    '''

    # Create a directed graph from the elements and relations
    G = nx.DiGraph()
    G.add_nodes_from(elements)
    G.add_edges_from(relations)

    closure_graph = nx.transitive_closure(G) # transitive closure
    poset_closure = set(closure_graph.edges) # edges

    return closure_graph, poset_closure