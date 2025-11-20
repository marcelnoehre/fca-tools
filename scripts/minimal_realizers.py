import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))

from src.naive.data import N5, M3
from src.naive.lattice import construct_closure
from src.naive.linear_extensions import all_linear_extensions, all_minimal_realizers
from src.utils.logger import log
from src.utils.constants import *

data = [('Lattice: N5', N5()), ('Lattice: M3', M3())]

for name, (elements, relations) in data:
    log(name, RED)
    closure_graph, poset_closure = construct_closure(elements, relations)
    linear_extensions = all_linear_extensions(poset_closure)

    print("Number of Linear Extensions:", len(linear_extensions))
    for ext in linear_extensions:
        log(ext, YELLOW)

    k, mle = all_minimal_realizers(linear_extensions, elements, poset_closure)

    print(f"Minimum Linear Extensions (k={k}):")
    for tuple in mle:
        log(tuple, GREEN)