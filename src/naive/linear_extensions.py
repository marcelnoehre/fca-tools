import networkx as nx

from typing import Set, Tuple, List
from itertools import combinations

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

def all_minimal_realizers(linear_extensions: List[List], elements: List, poset_closure: Set[Tuple]):
    '''
    Find all minimal realizers from a list of linear extensions.

    Args:
        linear_extensions (List[List]): A list of linear extensions, each represented as a list of elements.
        elements (List): A list of elements in the poset.
        poset_closure (Set[Tuple]): A set of tuples representing the order relations in the poset.

    Returns:
        Tuple[int, List[List]]: A tuple containing the size of the minimal realizer and a list of all minimal realizers.     
    '''

    def _intersection_of_orders(orders, elements):
        elements_list = list(elements)
        pair_all = set()
        
        for i in elements_list:
            for j in elements_list:
                if i == j:
                    continue
                # check if i < j in all orders
                if all(order.index(i) < order.index(j) for order in orders):
                    pair_all.add((i, j))

        _, poset_closure = construct_closure(elements_list, pair_all)
        
        return poset_closure
    
    for k in range(1, len(linear_extensions) + 1):
        valid_sets = [
            k_linear_extensions
            for k_linear_extensions in combinations(linear_extensions, k)
            if _intersection_of_orders(k_linear_extensions, elements) == poset_closure
        ]
        if valid_sets:
            return k, valid_sets
