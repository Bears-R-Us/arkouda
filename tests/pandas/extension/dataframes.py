import pandas as pd
import pytest

import arkouda as ak
from arkouda.pandas.extension._ArkoudaArray import ArkoudaArray
from arkouda.pandas.extension._ArkoudaCategoricalArray import ArkoudaCategoricalArray
from arkouda.pandas.extension._ArkoudaStringArray import ArkoudaStringArray


class TestDataFrameExtension:
    def test_dataframe_with_extensionarrays(self):
        N = 5
        df = pd.DataFrame(
            {
                "i": ArkoudaArray(ak.arange(N)),
                "s": ArkoudaStringArray(ak.array(["a", "b", "c", "d", "e"])),
                "c": ArkoudaCategoricalArray(
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

        # Convert to numpy for easy assertion
        # Avoid sort_values since it triggers iteration
        result_np = pd.DataFrame(
            {
                "id": result["id"].to_numpy(),
                "name": result["name"].to_numpy(),
                "score": result["score"].to_numpy(),
            }
        )
        result_np = result_np.sort_values("id", na_position="last").reset_index(drop=True)

        if how == "inner":
            assert len(result_np) == 2
            assert result_np["id"].tolist() == [2, 3]
            assert result_np["score"].tolist() == [88, 92]
        elif how == "left":
            assert len(result_np) == 3
            assert result_np["id"].tolist() == [1, 2, 3]
        elif how == "right":
            assert len(result_np) == 3
            assert result_np["id"].tolist() == [2, 3, 4]
        elif how == "outer":
            assert len(result_np) == 4
            assert sorted(result_np["id"].tolist()) == [1, 2, 3, 4]
