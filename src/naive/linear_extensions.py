from typing import Set, Tuple, List
import networkx as nx
from src.utils.constants import *
from src.naive.lattice import construct_closure

def all_linear_extensions(
        edges: Set[Tuple]
    ) -> List[List]:
    '''
    Generate all linear extensions of a poset defined by its edges using backtracking.

    Args:
        edges (Set[Tuple]): A set of tuples representing the order relations between elements.

    Returns:
        List[List]: A list of linear extensions, each represented as a list of elements.
    '''
    G = nx.DiGraph()
    G.add_edges_from(edges)

    def extend(partial):
        if len(partial) == len(G): # complete linear extension
            yield partial
        else:
            # nodes not in partial and whose predecessors are all in partial
            for n in [n for n in G.nodes if n not in partial and all(pred in partial for pred in G.predecessors(n))]:
                yield from extend(partial + [n])

    return list(extend([]))
