import matplotlib.pyplot as plt
from typing import List
from fcapy.lattice import ConceptLattice
from collections import defaultdict

class DimDraw2D():
    ''''
    Disclaimer:

    This Python script is a recreation of the algorithm originally developed by Prof. Dr. Dominik Dürrschnabel.
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
        self.grid = defaultdict(list)
        self.grid[self.realizer[0][-1]].append((0, 0))
        self.connections = []

        ext1, ext2 = (reversed(r[-1:] + r[:-1]) for r in self.realizer[:2])
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

    def draw(self):
        '''
        Draw the concept lattice using the assigned grid positions.
        '''
        plt.figure(figsize=(8, 6))
        # Grid
        for node, positions in self.grid.items():
            for pos in positions:
                plt.scatter([pos[0]], [pos[1]], color="black")
                plt.text(pos[0] - 0.2, pos[1] - 0.2, str(node), fontsize=12)

        for connection in self.connections:
            plt.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], color="black")

        plt.axis("equal")
        plt.axis("off")
        plt.savefig('dimdraw2d_lattice.png')
        plt.show()