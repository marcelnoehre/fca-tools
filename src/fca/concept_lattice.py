import networkx as nx
from itertools import chain, combinations
from collections import Counter, deque, defaultdict
from typing import List, Set, Tuple
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def concept_lattice(formal_context: FormalContext) -> ConceptLattice:
    return ConceptLattice.from_context(formal_context)

def count_linear_extensions(concept_lattice: ConceptLattice) -> int:
    counts = defaultdict(int) # initialize counts to 0
    counts[0] = 1 # top element gets count 1

    queue = deque(concept_lattice.children(0))
    while queue:
        node = queue.popleft()
        parents = concept_lattice.parents(node)

        if all(parent in counts for parent in parents):
            # sum up counts from parents
            counts[node] = sum(counts[parent] for parent in parents)
            
            # add children to queue
            children = concept_lattice.children(node)
            for child in children:
                if child not in queue:
                    queue.append(child)

        else:
            queue.append(node) # re-add to queue
    
    # bottom element holds count of linear extensions
    return counts[len(concept_lattice.to_networkx()) - 1]

def linear_extensions_topological(concept_lattice: ConceptLattice) -> List[List[int]]:
    return list(nx.all_topological_sorts(concept_lattice.to_networkx()))

def cover_relations(concept_lattice: ConceptLattice) -> Set[Tuple[int, int]]:
    return set(nx.transitive_reduction(concept_lattice.to_networkx()).edges)

def transitive_closure(concept_lattice: ConceptLattice) -> Set[Tuple[int, int]]:
    return set(nx.transitive_closure(concept_lattice.to_networkx()).edges)

def all_children(concept_lattice: ConceptLattice, index: int) -> Set[int]:
    visited = set()
    queue = deque([index])
    result = set()

    while queue:
        node = queue.popleft()
        for child in concept_lattice.children(node):
            if child not in visited:
                visited.add(child)
                result.add(child)
                queue.append(child)

    return result

def all_parents(concept_lattice: ConceptLattice, index: int) -> Set[int]:
    visited = set()
    queue = deque([index])
    result = set()

    while queue:
        node = queue.popleft()
        for parent in concept_lattice.parents(node):
            if parent not in visited:
                visited.add(parent)
                result.add(parent)
                queue.append(parent)

    return result

def intent_of_concept(concept_lattice: ConceptLattice, index: int) -> Set[int]:
    parents = all_parents(concept_lattice, index)
    intent = set()
    for feature in concept_lattice.get_concept_new_intent(index):
        intent.add((index, feature))
    for parent in parents:
        for feature in concept_lattice.get_concept_new_intent(parent):
            intent.add((parent, feature))
    return intent

def extent_of_concept(concept_lattice: ConceptLattice, index: int) -> Set[int]:
    children = all_children(concept_lattice, index)
    extent = set()
    for obj in concept_lattice.get_concept_new_extent(index):
        extent.add((index, obj))
    for child in children:
        for obj in concept_lattice.get_concept_new_extent(child):
            extent.add((child, obj))
            
    return extent

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def partial_linear_extensions(concept_lattice: ConceptLattice) -> List[List[int]]:
    G = concept_lattice.to_networkx()
    extensions = []
    for subset in powerset(G.nodes):
        subgraph = G.subgraph(subset).copy()
        extensions.extend(list(nx.all_topological_sorts(subgraph)))
    return extensions

def local_dimension(partial_realizers):
    return min(max(Counter(ple for ple in pr for ple in ple).values()) for pr in partial_realizers)

def relative_dimension(partial_realizers, elements):
    if not elements:
        return 0
    return min(sum(len(ple) for ple in pr) / len(elements) for pr in partial_realizers)