import matplotlib.pyplot as plt
from fcapy.visualizer import LineVizNx
from fcapy.lattice import ConceptLattice

def plot_concept_lattice_vsl(
        lat: ConceptLattice,
        filename: str
    ) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    vsl = LineVizNx()
    vsl.draw_concept_lattice(lat, ax=ax, flg_node_indices=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()