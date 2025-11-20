from src.utils.constants import *
from typing import List, Set, Tuple

def N5() -> Tuple[List, Set[Tuple]]:
    '''
    The N5 lattice.

          TOP
         /   \
        |     B
        A     |
        |     C
         \   /
          BOT

    Returns:
        elements: List of elements in the lattice.
        relations: Set of order relations between elements.
    '''
    elements = [TOP, A, B, C, BOT]
    relations = {(TOP, A), (TOP, B), (B, C), (A, BOT), (C, BOT)}
    return elements, relations

def M3() -> Tuple[List, Set[Tuple]]:
    '''
    The M3 lattice.

          TOP
        /  |  \
        A  B  C
        \  |  /
          BOT

    Returns:
        elements: List of elements in the lattice.
        relations: Set of order relations between elements.
    '''
    elements = [TOP, A, B, C, BOT]
    relations = {(TOP, A), (TOP, B), (TOP, C), (A, BOT), (B, BOT), (C, BOT)}
    return elements, relations