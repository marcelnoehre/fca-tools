import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.constants import *
from src.naive.linear_extensions import *
from src.fca.formal_context import *
from src.fca.concept_lattice import *

fc = formal_context(os.path.join(BASE_DIR, ANIMAL_MOVEMENT_PATH))
cl = concept_lattice(fc)
ple = partial_linear_extensions(cl)
k, min_sets = all_minimal_partial_realizers(ple, cl.to_networkx().nodes, transitive_closure(cl))
print(f'Minimal partial realizer size: {k}')
print(f'Local Dimension: {local_dimension(min_sets)}')
print(f'Relative Dimension: {relative_dimension(min_sets, cl.to_networkx().nodes)}')