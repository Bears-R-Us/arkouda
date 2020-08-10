import numpy as np
import pandas as pd
from context import arkouda as ak
from arkouda import dtypes
from base_test import ArkoudaTest

def getter(pd: pd.DataFrame, kname):

    akdf = ak.DFrame(pd)
    assert pd.Series(ak.to_akdf[kname].to_ndarray()) == pd[kname]





