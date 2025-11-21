import networkx as nx
from typing import List
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def concept_lattice(formal_context: FormalContext) -> ConceptLattice:
    return ConceptLattice.from_context(formal_context)

def linear_extensions(concept_lattice: ConceptLattice) -> List[List[int]]:
    return list(nx.all_topological_sorts(concept_lattice.to_networkx()))