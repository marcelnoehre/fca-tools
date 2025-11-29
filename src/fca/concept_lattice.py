import copy
import networkx as nx
from itertools import chain, combinations
from collections import Counter, deque, defaultdict
from typing import List, Set, Tuple
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def concept_lattice(formal_context: FormalContext) -> ConceptLattice:
    return ConceptLattice.from_context(formal_context)

def incomparability_graph(concept_lattice: ConceptLattice) -> nx.Graph:
    return nx.complement(nx.transitive_closure(concept_lattice.to_networkx()).to_undirected())

def count_linear_extensions(complement_concept_lattice: ConceptLattice) -> int:
    counts = defaultdict(int) # initialize counts to 0
    counts[0] = 1 # top element gets count 1

    queue = deque(complement_concept_lattice.children(0))
    while queue:
        node = queue.popleft()
        parents = complement_concept_lattice.parents(node)

        if all(parent in counts for parent in parents):
            # sum up counts from parents
            counts[node] = sum(counts[parent] for parent in parents)
            
            # add children to queue
            children = complement_concept_lattice.children(node)
            for child in children:
                if child not in queue:
                    queue.append(child)

        else:
            queue.append(node) # re-add to queue
    
    # bottom element holds count of linear extensions
    return counts[len(complement_concept_lattice.to_networkx()) - 1]

def all_linear_extensions(
        concept_lattice: ConceptLattice,
        complement_concept_lattice: ConceptLattice
    ) -> Set[List[int]]:

    def feature_chains(
            concept_lattice: ConceptLattice,
            intent_chain: List[List[Set[str]]] = [[]],
            node: int = 0,
        ) -> List[List[Set[str]]]:

        # add intent of current node to last chain
        intent_chain[len(intent_chain) - 1].append(intent_of_concept(concept_lattice, node).difference(*intent_chain[len(intent_chain) - 1]))
        
        children = concept_lattice.children(node)

        # follow single child chains
        while len(children) == 1:
            intent_chain[len(intent_chain) - 1].append(intent_of_concept(concept_lattice, list(children)[0]).difference(*intent_chain[len(intent_chain) - 1]))
            children = concept_lattice.children(list(children)[0])
        
        # branch on multiple children
        if len(children) > 1:
            # store current intent state
            intent_state = copy.deepcopy(intent_chain[-1])

            for i, child in enumerate(children):
                # for all but first child, create new chain starting from saved state
                if i != 0:
                    intent_chain.append(copy.deepcopy(intent_state))
                
                # recursively process child
                intent_chain = feature_chains(concept_lattice, intent_chain, child)

        return intent_chain
    
    intent_chains = feature_chains(complement_concept_lattice)

    linear_extensions = set()

    for intent_chain in intent_chains:
        # start with top element
        linear_extension = [len(list(concept_lattice.to_networkx().nodes)) - 1]
        intent_chain.pop(0) # remove top element intent

        # map nodes to the intents they introduce
        intent_dict = { node: concept_lattice.get_concept_new_intent(node) for node in concept_lattice.to_networkx().nodes }
        
        for intent in intent_chain:
            # find parent with matching intent
            node = next(k for k, v in intent_dict.items() if v == {x[1] for x in intent})
            linear_extension.append(node)

        linear_extensions.add(tuple(reversed(linear_extension + [0])))
    
    return linear_extensions

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