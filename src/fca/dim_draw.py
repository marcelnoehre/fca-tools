import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from fcapy.lattice import ConceptLattice
from collections import defaultdict, deque
from src.fca.concept_lattice import cover_relations
from src.utils.logger import log
from src.utils.constants import RED, YELLOW, GREEN

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
    
    def _plot_lattice(self,
            filename: str,
            nodes: Dict[int, Tuple[int, int]],
            relations: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            highlight_nodes: List[int] = []
        ):
        plt.figure(figsize=(8, 6))
        # Grid
        for node, positions in self.grid.items():
            for pos in positions:
                plt.scatter([pos[0]], [pos[1]], color="lightgrey", zorder=1)
                plt.text(pos[0] - 0.2, pos[1] - 0.2, node, fontsize=12, color='grey')
        for connection in self.connections:
            plt.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], color="lightgrey", zorder=1)

        # Nodes
        for node, coordinate in nodes.items():
            plt.scatter([coordinate[0]], [coordinate[1]], color="orange" if node in highlight_nodes else "blue", zorder=3)
            plt.text(coordinate[0], coordinate[1] + 0.3, self.concept_lattice.get_concept_new_intent(node), fontsize=12, ha='center', va='top', color='grey')
            plt.text(coordinate[0], coordinate[1] - 0.3, self.concept_lattice.get_concept_new_extent(node), fontsize=12, ha='center', va='bottom', color='grey')

        # Relations
        for relation in relations:
            plt.plot([relation[0][0], relation[1][0]], [relation[0][1], relation[1][1]], color="black", zorder=2)

        plt.axis("equal")
        plt.axis("off")
        plt.savefig(filename)
        plt.show()

    def _irreducible_nodes(self):
        self.join_irreducibles = list(reversed([node for node, child in self.concept_lattice.children_dict.items() if len(child) == 1]))
        self.meet_irreducibles = [node for node, parent in self.concept_lattice.parents_dict.items() if len(parent) == 1]

    def _bottom_up_additive(self):
        self.bottom_up_additive = {}
        q = deque()

        root = self.realizer[0][-1]
        self.bottom_up_additive[root] = self.nodes[root]
        q.append(root)

        for node in self.join_irreducibles:
            self.bottom_up_additive[node] = self.nodes[node]
            q.append(node)

        children = self.concept_lattice.children_dict
        remaining = set(list(self.concept_lattice.to_networkx().nodes)) - set(self.bottom_up_additive)

        while remaining:
            for node in list(remaining):
                childs = list(children[node])
                if all(c in self.bottom_up_additive for c in childs):
                    x = 0
                    y = 0
                    for c in childs:
                        cx, cy = self.bottom_up_additive[c]
                        x += cx
                        y += cy
                    self.bottom_up_additive[node] = (x, y)
                    q.append(node)
                    remaining.remove(node)

    def _top_down_additive(self):
        self.top_down_additive = {}
        q = deque()

        root = self.realizer[0][0]
        self.top_down_additive[root] = self.nodes[root]
        q.append(root)

        for node in self.meet_irreducibles:
            self.top_down_additive[node] = self.nodes[node]
            q.append(node)

        parents = self.concept_lattice.parents_dict
        remaining = set(list(self.concept_lattice.to_networkx().nodes)) - set(self.top_down_additive)

        while remaining:
            for node in list(remaining):
                pars = list(parents[node])
                if all(p in self.top_down_additive for p in pars):
                    x = 0
                    y = 0
                    for p in pars:
                        px, py = self.top_down_additive[p]
                        x += px
                        y += py
                    ox, oy = self.nodes[root]
                    self.top_down_additive[node] = (x - ox, y - oy)
                    q.append(node)
                    remaining.remove(node)

    def _combined_additive(self):
        self.combined_additive = {}
        for node in self.concept_lattice.to_networkx().nodes:
            self.combined_additive[node] = (
                self.bottom_up_additive[node][0] + self.top_down_additive[node][0],
                self.bottom_up_additive[node][1] + self.top_down_additive[node][1]
            )

        top = min(self.nodes.keys())
        bottom = max(self.nodes.keys())

        pos_original_top = np.array(self.nodes[top], dtype=float)
        pos_original_bottom = np.array(self.nodes[bottom], dtype=float)

        pos_target_top = np.array(self.combined_additive[top], dtype=float)
        pos_target_bottom = np.array(self.combined_additive[bottom], dtype=float)

        scale = (pos_original_bottom - pos_original_top) / (pos_target_bottom - pos_target_top)
        t = pos_original_top - scale * pos_target_top

        def _transform(point):
            v = np.array(point, dtype=float)
            return tuple(scale * v + t)
        
        self.scaled_combined_additive = { k: _transform(v) for k, v in self.combined_additive.items() }

    def plot_dim_draw(self):
        '''
        Plot the concept lattice using the dim draw approach.
        '''
        relations = [(self.nodes[a], self.nodes[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("dim_draw.png", self.nodes, relations)

    def plot_bottom_up_additive(self):
        relations = [(self.bottom_up_additive[a], self.bottom_up_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("bottom_up_additive.png", self.bottom_up_additive, relations, self.join_irreducibles)

    def plot_top_down_additive(self):
        relations = [(self.top_down_additive[a], self.top_down_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("top_down_additive.png", self.top_down_additive, relations, self.meet_irreducibles)

    def plot_combined_additive(self):
        relations = [(self.combined_additive[a], self.combined_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("combined_additive.png", self.combined_additive, relations)

    def plot_scaled_combined_additive(self):
        relations = [(self.scaled_combined_additive[a], self.scaled_combined_additive[b]) for a, b in cover_relations(self.concept_lattice)]
        self._plot_lattice("scaled_combined_additive.png", self.scaled_combined_additive, relations)

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

        def check(label, values):
            additive = True
            log(f'{label}', YELLOW)
            to_str=lambda v: f"{float(v[0]):.2f}, {float(v[1]):.2f}"
            for node in self.concept_lattice.to_networkx().nodes:
                if values[node] != self.nodes[node]:
                    additive = False
                    log(f"Not {f'{label} additive'.lower()} at node {node}: expected {to_str(self.nodes[node])}, got {to_str(values[node])}", RED)
            if additive:
                log(f"The DimDraw drawing is {f'{label} additive'.lower()}", GREEN)
            return additive

        self.check_bottom_up_additive = check('Bottom Up', self.bottom_up_additive)
        self.check_top_down_additive = check('Top Down', self.top_down_additive)
        self.check_combined_additive = check('Combined', self.scaled_combined_additive)
