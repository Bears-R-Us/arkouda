import pandas as pd
import pytest

import arkouda as ak

from arkouda.pandas.extension._arkouda_array import ArkoudaArray
from arkouda.pandas.extension._arkouda_categorical_array import ArkoudaCategorical
from arkouda.pandas.extension._arkouda_string_array import ArkoudaStringArray


class TestDataFrameExtension:
    def test_dataframe_with_extensionarrays(self):
        N = 5
        df = pd.DataFrame(
            {
                "i": ArkoudaArray(ak.arange(N)),
                "s": ArkoudaStringArray(ak.array(["a", "b", "c", "d", "e"])),
                "c": ArkoudaCategorical(
                    ak.Categorical(ak.array(["low", "low", "high", "medium", "low"]))
                ),
            }
        )

        assert df.shape == (5, 3)
        assert df.iloc[2]["s"] == "c"

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_dataframe_join_with_arkouda_arrays(self, how):
        # Left table
        df1 = pd.DataFrame(
            {
                "id": ArkoudaArray(ak.array([1, 2, 3])),
                "name": ArkoudaStringArray(ak.array(["alice", "bob", "carol"])),
            }
        )

        # Right table
        df2 = pd.DataFrame(
            {
                "id": ArkoudaArray(ak.array([2, 3, 4])),
                "score": ArkoudaArray(ak.array([88, 92, 75])),
            }
        )

        # Perform join
        result = df1.merge(df2, on="id", how=how)

        if how == "inner":
            assert len(result) == 2
            assert result["id"].tolist() == [2, 3]
            assert result["score"].tolist() == [88, 92]
        elif how == "left":
            assert len(result) == 3
            assert result["id"].tolist() == [1, 2, 3]
        elif how == "right":
            assert len(result) == 3
            assert result["id"].tolist() == [2, 3, 4]
        elif how == "outer":
            assert len(result) == 4
            assert sorted(result["id"].tolist()) == [1, 2, 3, 4]
