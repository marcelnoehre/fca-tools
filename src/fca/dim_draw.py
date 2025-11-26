import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from fcapy.lattice import ConceptLattice
from collections import defaultdict, deque
from src.fca.concept_lattice import *
from src.utils.logger import log
from src.utils.constants import RED, YELLOW, GREEN, MAGENTA, CYAN, RESET

class DimDraw2D():
    ''''
    Disclaimer:

    This Python script is based on the DimDraw originally developed by Prof. Dr. Dominik Dürrschnabel.
    This code is an independent visualization and may differ from the original algorithm in structure and performance.
    The original algorithm is integrated into the tool conexp-clj [see https://github.com/tomhanika/conexp-clj].

    @misc{dürrschnabel2019dimdrawnoveltool,
        title={DimDraw -- A novel tool for drawing concept lattices},
        author={Dominik Dürrschnabel and Tom Hanika and Gerd Stumme},
        year={2019},
        eprint={1903.00686},
        archivePrefix={arXiv},
        primaryClass={cs.CG},
        url={https://arxiv.org/abs/1903.00686}
    }
    '''

    def __init__(self,
            concept_lattice: ConceptLattice,
            realizer: List[List[int]]
        ):
        '''
        Initialize DimDraw2D with a given realizer.

        Args:
            concept_lattice (ConceptLattice): The concept lattice to be visualized.
            realizer (List[List[int]]): A list of linear extensions forming the realizer.
        '''
        self.concept_lattice = concept_lattice
        self.realizer = realizer
        self.objects = extent_of_concept(self.concept_lattice, self.realizer[0][0])
        self.features = intent_of_concept(self.concept_lattice, self.realizer[0][-1])
        self._setup_grid()

    def _setup_grid(self):
        self.n = len(self.concept_lattice.to_networkx().nodes)
        self.grid = defaultdict(list)
        self.grid[self.realizer[0][-1]].append((0, 0))
        self.connections = []

        ext1, ext2 = (reversed(r[-1:] + r[1:-1]) for r in self.realizer[:2])
        prev_x, prev_y = (0, 0)

        for i, (node_x, node_y) in enumerate(zip(ext1, ext2)):
            # horizontal
            self.grid[node_x].append((i + 1, 0))
            self.connections.append(((prev_x, 0), (i + 1, 0)))
            prev_x = i + 1
            # vertical
            self.grid[node_y].append((0, i + 1))
            self.connections.append(((0, prev_y), (0, i + 1)))
            prev_y = i + 1

        ext1, ext2 = (reversed(r[:-1]) for r in self.realizer[:2])
        self.connections.append(((self.n - 1, 0), (self.n - 1, 1)))
        self.connections.append(((0, self.n - 1), (1, self.n - 1)))
        for i, (node_x, node_y) in enumerate(zip(ext1, ext2)):
            # horizontal
            self.grid[node_x].append((i + 1, self.n - 1))
            self.connections.append(((prev_x, self.n - 1), (i + 1, self.n - 1)))
            prev_x = i + 1
            # vertical
            self.grid[node_y].append((self.n - 1, i + 1))
            self.connections.append(((self.n - 1, prev_y), (self.n - 1, i + 1)))
            prev_y = i + 1  

        self.nodes = dict(self.grid)
        self.nodes.pop(self.realizer[0][-1])
        self.nodes = {node: coords if node == 0 else [coord for coord in coords if 0 in coord] for node, coords in self.nodes.items()}
        self.nodes = {
            node: (coords[1][0], coords[0][1]) if coords[0][0] == 0 else (coords[0][0], coords[1][1])
            for node, coords in self.nodes.items()
        }
        self.nodes[self.realizer[0][-1]] = (0, 0)
        self._legend()
    
    def _legend(self):
        for node in self.concept_lattice.to_networkx().nodes:
            log(f'Node {node}:', MAGENTA)
            print(f'New extent: {YELLOW}{self.concept_lattice.get_concept_new_extent(node)}{RESET}')
            print(f'New intent: {YELLOW}{self.concept_lattice.get_concept_new_intent(node)}{RESET}')

    def _plot_lattice(self,
            filename: str,
            nodes: Dict[int, Tuple[int, int]],
            relations: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            highlight_nodes: List[int] = [],
            args: Optional[Dict[str, bool]] = None
        ):
        if args is None:
            args = {}
        for key in ['grid', 'concepts']:
            if key not in args:
                args[key] = False

        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
        
        plt.figure(figsize=(8, 6))
        # Grid
        if args['grid']:
            for node, positions in self.grid.items():
                for pos in positions:
                    x, y = R @ np.array(pos)
                    plt.scatter(x, y, color="lightgrey", zorder=1)
                    plt.text(x - 0.2, y - 0.2, node, fontsize=12, color='grey')
            for connection in self.connections:
                x0, y0 = R @ np.array(connection[0])
                x1, y1 = R @ np.array(connection[1])
                plt.plot([x0, x1], [y0, y1], color="lightgrey", zorder=1)

        # Nodes
        for node, coordinate in nodes.items():
            x, y = R @ np.array(coordinate)
            plt.scatter(x, y, color="orange" if node in highlight_nodes else "blue", zorder=3)
            if args['concepts']:
                plt.text(x, y + 0.5, self.concept_lattice.get_concept_new_extent(node), fontsize=12, ha='center', va='top', color='grey')
                plt.text(x, y - 0.5, self.concept_lattice.get_concept_new_intent(node), fontsize=12, ha='center', va='bottom', color='grey')

        # Relations
        for relation in relations:
            x0, y0 = R @ np.array(relation[0])
            x1, y1 = R @ np.array(relation[1])
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        plt.axis("equal")
        plt.axis("off")
        plt.savefig(filename)
        plt.show()

    def _irreducible_nodes(self):
        # all nodes with exactly one child
        join_irreducibles = list(reversed([node for node, child in self.concept_lattice.children_dict.items() if len(child) == 1]))
        self.join_irreducibles = {}
        for node in join_irreducibles:
            # vector from child to parent
            self.join_irreducibles[node] = (
                self.nodes[node][0] - self.nodes[list(self.concept_lattice.children_dict[node])[0]][0],
                self.nodes[node][1] - self.nodes[list(self.concept_lattice.children_dict[node])[0]][1]
            )

        # all nodes with exactly one parent
        meet_irreducibles = [node for node, parent in self.concept_lattice.parents_dict.items() if len(parent) == 1]
        self.meet_irreducibles = {}
        for node in meet_irreducibles:
            # vector from parent to child
            self.meet_irreducibles[node] = (
                self.nodes[list(self.concept_lattice.parents_dict[node])[0]][0] - self.nodes[node][0],
                self.nodes[list(self.concept_lattice.parents_dict[node])[0]][1] - self.nodes[node][1]
            )

    def _bottom_up_additive(self):
        # root = bottom element
        root = self.realizer[0][-1]
        
        # base vectors:
        # root node as (0, 0)
        # vector if join-irreducibles
        # sum of join-irreducibles for other nodes
        self.base_vectors_bottom = copy.deepcopy(self.join_irreducibles)
        self.base_vectors_bottom[root] = self.nodes[root]

        self.bottom_up_additive = {}
        self.bottom_up_additive[root] = self.nodes[root]

        queue = deque(self.concept_lattice.parents(root))
        while queue:
            node = queue.popleft()
            children = all_children(self.concept_lattice, node)
            # all children have to be processed first
            if all(child in self.bottom_up_additive for child in children):
                if node in self.join_irreducibles:
                    # base vector of join-irreducible + base vector of the single child
                    self.bottom_up_additive[node] = (
                        self.base_vectors_bottom[node][0] + self.base_vectors_bottom[list(self.concept_lattice.children(node))[0]][0],
                        self.base_vectors_bottom[node][1] + self.base_vectors_bottom[list(self.concept_lattice.children(node))[0]][1]
                    )

                else:
                    px, py = 0, 0
                    for child in all_children(self.concept_lattice, node):
                        # sum base vectors of all join-irreducible children
                        if child in self.join_irreducibles:
                            px += self.base_vectors_bottom[child][0]
                            py += self.base_vectors_bottom[child][1]
                    
                    # add sum as base vector for further nodes depending on this node
                    self.base_vectors_bottom[node] = (px, py)
                    self.bottom_up_additive[node] = (px, py)

                # add parents if not already processed or in queue
                for p in self.concept_lattice.parents(node):
                    if p not in queue and p not in self.bottom_up_additive:
                        queue.append(p)

            else:
                # re-add to queue if children not processed yet
                queue.append(node)
            
    def _top_down_additive(self):
        # root = top element
        root = self.realizer[0][0]

        # base vectors:
        # root node as (0, 0)
        # vector if meet-irreducibles
        # sum of meet-irreducibles for other nodes
        self.base_vectors_top = copy.deepcopy(self.meet_irreducibles)
        self.base_vectors_top[root] = (0, 0)

        self.top_down_additive = {}
        self.top_down_additive[root] = self.nodes[root]

        queue = deque(self.concept_lattice.children(root))
        while queue:
            node = queue.popleft()
            parents = all_parents(self.concept_lattice, node)
            # all parents have to be processed first
            if all(parent in self.top_down_additive for parent in parents):
                if node in self.meet_irreducibles:
                    # top node - (base vector of meet-irreducible + base vector of the single parent)
                    # ensures a positive base vector from parent to child
                    self.top_down_additive[node] = (
                        self.nodes[root][0] - (self.base_vectors_top[node][0] + self.base_vectors_top[list(self.concept_lattice.parents(node))[0]][0]),
                        self.nodes[root][1] - (self.base_vectors_top[node][1] + self.base_vectors_top[list(self.concept_lattice.parents(node))[0]][1])
                    )

                else:
                    px, py = 0, 0
                    for parent in parents:
                        if parent in self.meet_irreducibles:
                            # sum base vectors of all meet-irreducible parents
                            px += self.base_vectors_top[parent][0]
                            py += self.base_vectors_top[parent][1]

                    # add sum as base vector for further nodes depending on this node
                    self.base_vectors_top[node] = (px, py)
                    self.top_down_additive[node] = (self.nodes[root][0] - px, self.nodes[root][1] - py)
                
                # add children if not already processed or in queue
                for p in self.concept_lattice.children(node):
                    if p not in queue and p not in self.top_down_additive:
                        queue.append(p)

            else:
                # re-add to queue if parents not processed yet
                queue.append(node)
                
    def _combined_position(self, intent, extent):
        # bottom element as initial position
        pos = (0, 0)

        # sum up base vectors 
        for feature in self.features:
            # M \ B
            # all features not in the concept's intent
            if feature in intent:
                continue
            pos = (
                pos[0] + self.base_vectors_top[feature[0]][0],
                pos[1] + self.base_vectors_top[feature[0]][1]
            )

        # sum up base vectors 
        for object in extent:
            # A
            # all objects in the concept's extent 
            # only consider join-irreducibles
            if object[0] not in self.join_irreducibles.keys():
                continue
            pos = (
                pos[0] + self.base_vectors_bottom[object[0]][0],
                pos[1] + self.base_vectors_bottom[object[0]][1]
            )

        # final position of a node in the combined additive drawing
        return pos

    def _combined_additive(self):
        # root = bottom element
        root = self.realizer[0][-1]

        self.combined_additive = {}
        self.combined_additive[root] = self.nodes[root]

        # combined position for the root node
        self.combined_additive[root] = self._combined_position(intent_of_concept(self.concept_lattice, root), extent_of_concept(self.concept_lattice, root))

        queue = deque(self.concept_lattice.parents(root))
        while queue:
            node = queue.popleft()
            # combined position for the node
            self.combined_additive[node] = self._combined_position(intent_of_concept(self.concept_lattice, node), extent_of_concept(self.concept_lattice, node))
            
            # add parents if not already processed or in queue
            for p in self.concept_lattice.parents(node):
                if p not in queue and p not in self.combined_additive:
                    queue.append(p)

    def _scale_combined_additive(self):
        top = min(self.nodes.keys())
        bottom = max(self.nodes.keys())

        pos_original_top = np.array(self.nodes[top], dtype=float)
        pos_original_bottom = np.array(self.nodes[bottom], dtype=float)

        pos_combined_top = np.array(self.combined_additive[top], dtype=float)
        pos_combined_bottom = np.array(self.combined_additive[bottom], dtype=float)

        # scale = target_distance (original) / current_distance (combined) 
        scale = (pos_original_bottom - pos_original_top) / (pos_combined_bottom - pos_combined_top)
        # use minimum scale to maintain aspect ratio
        t = pos_original_top - scale * pos_combined_top

        def _transform(point):
            # transform point based on scale and translation
            v = np.array(point, dtype=float)
            return tuple(scale * v + t)
        
        self.scaled_combined_additive = { k: _transform(v) for k, v in self.combined_additive.items() }

    def plot_dim_draw(self, args: Optional[Dict[str, bool]] = None):
        '''
        Plot the concept lattice using the dim draw approach.
        '''
        relations = [(self.nodes[a], self.nodes[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("dim_draw.png", self.nodes, relations, [], args)

    def plot_bottom_up_additive(self, args: Optional[Dict[str, bool]] = None):
        relations = [(self.bottom_up_additive[a], self.bottom_up_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("bottom_up_additive.png", self.bottom_up_additive, relations, self.join_irreducibles, args)

    def plot_top_down_additive(self, args: Optional[Dict[str, bool]] = None):
        relations = [(self.top_down_additive[a], self.top_down_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("top_down_additive.png", self.top_down_additive, relations, self.meet_irreducibles, args)

    def plot_combined_additive(self, args: Optional[Dict[str, bool]] = None):
        relations = [(self.combined_additive[a], self.combined_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("combined_additive.png", self.combined_additive, relations, [], args)

    def plot_scaled_combined_additive(self, args: Optional[Dict[str, bool]] = None):
        relations = [(self.scaled_combined_additive[a], self.scaled_combined_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("scaled_combined_additive.png", self.scaled_combined_additive, relations, [], args)

    def check_additivity(self) -> bool:
        '''
        Check if the current drawing is additive.

        Returns:
            bool: True if the drawing is additive, False otherwise.
        '''
        self._irreducible_nodes()
        self._bottom_up_additive()
        self._top_down_additive()
        self._combined_additive()
        self._scale_combined_additive()

        def check(label, values):
            additive = True
            log(f'{label}', CYAN)
            to_str=lambda v: f"{float(v[0]):.2f}, {float(v[1]):.2f}"
            for node in self.concept_lattice.to_networkx().nodes:
                # node breaks additivity if positions differ more than the tolerance (1e-6)
                if abs(values[node][0] - self.nodes[node][0]) > 1e-6 or abs(values[node][1] - self.nodes[node][1]) > 1e-6:
                    additive = False
                    log(f"Not {f'{label}'.lower()} at node {YELLOW}{node}{RED}: expected {YELLOW}({to_str(self.nodes[node])}){RED}, got {YELLOW}({to_str(values[node])})", RED)
            
            if additive:
                log(f"The DimDraw drawing is {f'{label} additive'.lower()}", GREEN)
            
            return additive

        self.check_bottom_up_additive = check('Bottom Up Additive', self.bottom_up_additive)
        self.check_top_down_additive = check('Top Down Additive', self.top_down_additive)
        self.check_combined_additive = check('Combined Additive', self.scaled_combined_additive)
