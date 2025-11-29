import pandas as pd
from fcapy.context import FormalContext

def formal_context(path: str) -> FormalContext:
    return FormalContext.from_pandas(pd.read_csv(path, index_col=0))

def reverse_formal_context(fc: FormalContext) -> FormalContext:
    return FormalContext.from_pandas(~fc.to_pandas())