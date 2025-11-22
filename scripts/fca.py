import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.constants import ANIMAL_MOVEMENT_PATH, N5_PATH
from src.naive.linear_extensions import all_minimal_realizers
from src.fca.formal_context import *
from src.fca.concept_lattice import *
from src.fca.plot import plot_concept_lattice_vsl
from src.fca.dim_draw import DimDraw2D

fc = formal_context(os.path.join(BASE_DIR, N5_PATH))
cl = concept_lattice(fc)
k, realizers = all_minimal_realizers(linear_extensions(cl), cl.to_networkx().nodes, transitive_closure(cl))
dim_draw = DimDraw2D(cl, realizer=realizers[0])
dim_draw.plot_dim_draw()
dim_draw.check_additivity()
dim_draw.plot_bottom_up_additive()
dim_draw.plot_top_down_additive()
dim_draw.plot_combined_additive()
dim_draw.plot_scaled_combined_additive()