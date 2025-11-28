import copy
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, sympify
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
            else:
                plt.text(x - 0.2, y - 0.2, node, fontsize=12, ha='center', va='bottom', color='grey')

        # Relations
        for relation in relations:
            x0, y0 = R @ np.array(relation[0])
            x1, y1 = R @ np.array(relation[1])
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
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
        '''
        X := G
        (A, B) -> A 
        '''
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
                    child = self.concept_lattice.children(node)
                    child_vector = self.base_vectors_bottom[list(child)[0]]
                    # if the child is a join-irreducible, sum up the chain until a non-join-irreducible is found
                    while list(child)[0] in self.join_irreducibles:
                        child = self.concept_lattice.children(list(child)[0])
                        child_vector = (
                            child_vector[0] + self.base_vectors_bottom[list(child)[0]][0],
                            child_vector[1] + self.base_vectors_bottom[list(child)[0]][1]
                        )

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
        '''
        X := M
        (A, B) -> M \ B
        '''
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
                    parent = self.concept_lattice.parents(node)
                    parent_vector = self.base_vectors_top[list(parent)[0]]
                    
                    # if the parent is a meet-irreducible, sum up the chain until a non-meet-irreducible is found
                    while list(parent)[0] in self.meet_irreducibles:
                        parent = self.concept_lattice.parents(list(parent)[0])
                        parent_vector = (
                            parent_vector[0] + self.base_vectors_top[list(parent)[0]][0],
                            parent_vector[1] + self.base_vectors_top[list(parent)[0]][1]
                        )

                    # top node - (base vector of meet-irreducible + base vector of the single parent)
                    # ensures a positive base vector from parent to child
                    self.top_down_additive[node] = (
                        self.nodes[root][0] - (self.base_vectors_top[node][0] + parent_vector[0]),
                        self.nodes[root][1] - (self.base_vectors_top[node][1] + parent_vector[1])
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

    def _solve(self, prefix, node, solution_list, free_vars, vector_variables, expected):
        eq = [] # construct equation to solve for the current node
        for var in [s for group in self.base_vectors_combined[node] for (_, s) in group]:
            var = f'{prefix}{var}'
            if var in vector_variables:
                eq.append(str(vector_variables[var]))
            elif symbols(var) in free_vars:
                eq.append(str(var))
            else:
                sub_eq = str(solution_list[0][symbols(var)])
                for k, v in vector_variables.items():
                    sub_eq = sub_eq.replace(str(k), str(v))
                if not any(var in sub_eq for var in self.variables):
                    vector_variables[var] = int(eval(sub_eq, {"__builtins__": None}))
                    sub_eq = str(vector_variables[var])
                eq.append(f'({sub_eq})')

        eq = ' + '.join(eq)

        # if the equation contains variables, solve it
        if any(var in eq for var in self.variables):
            # define sympy variables and equations
            expr_vars = [v for v in self.variables if eq.find(v) != -1]
            expr_symbols = symbols(' '.join(expr_vars))
            expr = Eq(sympify(eq), expected) 
            solution = solve(expr, expr_symbols, dict=True)
            
            # if a solution is found, update the variable values
            if solution:
                for k, v in solution[0].items():
                    vector_variables[str(k)] = int(v)
            
            # if no solution is found the variables cancel out -> assign 0 to all variables
            else:
                for var in expr_vars:
                    if str(var) not in list(vector_variables.keys()):
                        vector_variables[str(var)] = 0

        # updated vector variables
        return vector_variables
                
    def _combined_additive(self):
        '''
        X := G U M
        (A, B) -> A U (M \ B)
        '''
        # root = bottom element
        root = self.realizer[0][-1]

        self.variables = [f'{prefix}_{v}' for prefix in ('x', 'y') for _, v in list(self.objects) + list(self.features)]
        self.equations = []
        self.base_vectors_combined = {}

        # vectors for the root node
        extent = extent_of_concept(self.concept_lattice, root)
        complement_intent = [f for f in self.features if f not in intent_of_concept(self.concept_lattice, root)]
        self.base_vectors_combined[root] = (list(extent), complement_intent)
        
        # linear equations for the root node
        variables = [v for _, v in extent] + [v for _, v in complement_intent]
        self.equations.append((' + '.join(f'x_{v}' for v in variables) if variables else '0') + f' = {self.nodes[root][0]}')
        self.equations.append((' + '.join(f'y_{v}' for v in variables) if variables else '0') + f' = {self.nodes[root][1]}')

        queue = deque(self.concept_lattice.parents(root))
        while queue:
            node = queue.popleft()
            # vectors for the node
            extent = extent_of_concept(self.concept_lattice, node)
            complement_intent = [f for f in self.features if f not in intent_of_concept(self.concept_lattice, node)]
            self.base_vectors_combined[node] = (list(extent), complement_intent)

            # linear equations for the node
            variables = [v for _, v in extent] + [v for _, v in complement_intent]
            self.equations.append((' + '.join(f'x_{v}' for v in variables) if variables else '0') + f' = {self.nodes[node][0]}')
            self.equations.append((' + '.join(f'y_{v}' for v in variables) if variables else '0') + f' = {self.nodes[node][1]}')
            
            # add parents if not already processed or in queue
            for p in self.concept_lattice.parents(node):
                if p not in queue and p not in self.base_vectors_combined:
                    queue.append(p)

        # define sympy variables and equations
        vars = symbols(' '.join(self.variables))
        eqs = []
        for eq in self.equations:
            left, right = eq.split('=')
            eqs.append(Eq(sympify(left), sympify(right)))

        # solve equations
        solution_list = solve(eqs, vars, dict=True)
        
        # identify free variables that can take any value
        free_vars = [v for v in vars if v not in solution_list[0]]

        # extract already fixed variable values
        vector_variables = {}
        for k, v in solution_list[0].items():
            try:
                vector_variables[str(k)] = int(v)
            except TypeError:
                continue

        # add root node
        self.combined_additive = {}
        pos = (0, 0)
        for index, prefix in enumerate(['x_', 'y_']):
            # solve until all variables for the current node are known
            while not all(f'{prefix}{var}' in vector_variables.keys() for var in [s for group in self.base_vectors_combined[root] for (_, s) in group]):
                vector_variables = self._solve(prefix, root, solution_list, free_vars, vector_variables, self.nodes[root][index])
        
            # sum up all base vectors for the root node
            # index defines the direction (0 = x, 1 = y)
            for vec in [s for group in self.base_vectors_combined[root] for (_, s) in group]:
                pos = (
                    pos[0] + vector_variables[f'{prefix}{vec}'] if index == 0 else pos[0],
                    pos[1] + vector_variables[f'{prefix}{vec}'] if index == 1 else pos[1]
                )

        self.combined_additive[root] = pos

        queue = deque(self.concept_lattice.parents(root))
        while queue:
            node = queue.popleft()

            pos = (0, 0) # bottom up approach
            for index, prefix in enumerate(['x_', 'y_']):
                # solve until all variables for the current node are known
                while not all(f'{prefix}{var}' in vector_variables.keys() for var in [s for group in self.base_vectors_combined[node] for (_, s) in group]):
                    vector_variables = self._solve(prefix, node, solution_list, free_vars, vector_variables, self.nodes[node][index])
            
                # sum up all base vectors for the current node
                # index defines the direction (0 = x, 1 = y)
                for vec in [s for group in self.base_vectors_combined[node] for (_, s) in group]:
                    pos = (
                        pos[0] + vector_variables[f'{prefix}{vec}'] if index == 0 else pos[0],
                        pos[1] + vector_variables[f'{prefix}{vec}'] if index == 1 else pos[1]
                    )

            self.combined_additive[node] = pos

            # add parents if not already processed or in queue
            for p in self.concept_lattice.parents(node):
                if p not in queue and p not in self.combined_additive:
                    queue.append(p)

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
        self.check_combined_additive = check('Combined Additive', self.combined_additive)
