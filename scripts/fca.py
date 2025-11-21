import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.constants import N5_PATH
from src.naive.linear_extensions import all_minimal_realizers
from src.fca.formal_context import *
from src.fca.concept_lattice import *
from src.fca.plot import plot_concept_lattice_vsl

fc = formal_context(os.path.join(BASE_DIR, N5_PATH))
cl = concept_lattice(fc)
plot_concept_lattice_vsl(cl, 'n5_lattice.png')
k, realizers = all_minimal_realizers(linear_extensions(cl), cl.to_networkx().nodes, transitive_closure(cl))
print(f'Minimal realizer size: {k}')
for realizer in realizers:
    print(realizer)