from unittest import result
import networkx as nx
from collections import deque
from typing import List, Set, Tuple
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def concept_lattice(formal_context: FormalContext) -> ConceptLattice:
    return ConceptLattice.from_context(formal_context)

def linear_extensions(concept_lattice: ConceptLattice) -> List[List[int]]:
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