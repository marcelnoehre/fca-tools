import networkx as nx
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
