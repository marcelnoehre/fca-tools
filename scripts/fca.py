import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.constants import N5_PATH
from src.fca.formal_context import formal_context

fc = formal_context(os.path.join(BASE_DIR, N5_PATH))
print(fc)