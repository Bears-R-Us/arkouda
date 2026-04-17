"""
Minimal IO tests for **pd.Index backed by ArkoudaExtensionArray**.

Goal
----
Only verify that a pandas `Index` whose underlying data is an Arkouda-backed
ExtensionArray can round-trip through:

- `idx.ak.to_csv(...)`
- `idx.ak.to_parquet(...)`
- `idx.ak.to_hdf(...)`

These tests intentionally avoid `ak.Index` / `ak.MultiIndex` and focus on the
pandas surface area (Index + `.ak` accessor).
"""

from __future__ import annotations

import os
import tempfile

from typing import Any

import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda.pandas import io_util
from arkouda.pandas.extension import ArkoudaArray


def _make_pd_index_with_arkouda_ea() -> pd.Index:
    """
    Construct a pd.Index backed by ArkoudaExtensionArray.

    We intentionally go through `pd.array(ak_array)` because that is the
    canonical way pandas creates an ExtensionArray, and Arkouda's integration
    should return an Arkouda-backed EA here.
    """
    ak_arr = ArkoudaArray(ak.arange(10))
    ea = pd.array(ak_arr)  # should be ArkoudaExtensionArray (or equivalent)
    idx = pd.Index(ea, name="my_index")
    return idx


def _assert_is_arkouda_backed_index(idx: pd.Index) -> None:
    # Must expose `.ak` accessor for IO calls.
    assert hasattr(idx, "ak"), "Expected pandas Index to have an `.ak` accessor"

    # Must be ExtensionArray-backed (not numpy).
    arr = idx.array
    assert not isinstance(arr, np.ndarray), (
        "Expected Index to be backed by an ExtensionArray, not ndarray"
    )


def _read_index_from_csv(path: str):
    data = ak.read_csv(path)
    assert isinstance(data, dict)
    assert len(data) >= 1
    # Prefer the conventional "Index" column name if present.
    return data.get("Index", next(iter(data.values())))


def _read_index_from_parquet(path_glob: str):
    data = ak.read_parquet(path_glob)
    assert isinstance(data, dict)
    assert len(data) >= 1
    return data.get("Index", next(iter(data.values())))


def _read_index_from_hdf(path_glob: str):
    data = ak.read_hdf(path_glob)
    assert isinstance(data, dict)
    assert len(data) >= 1
    return data.get("Index", next(iter(data.values())))


def _assert_roundtrip_equal(expected: pd.Index, got: Any) -> None:
    # `ak.read_*` returns arkouda objects (pdarray/Strings/etc). Normalize to ndarray.
    if hasattr(got, "to_ndarray"):
        got_np = got.to_ndarray()
    else:
        got_np = np.asarray(got)

    np.testing.assert_array_equal(expected.to_numpy(), got_np)


@pytest.fixture
def index_io_tmp(request):
    base = f"{pytest.temp_directory}/.pd_index_ea_io_test"
    io_util.get_directory(base)

    def finalizer():
        io_util.delete_directory(base)

    request.addfinalizer(finalizer)
    return base


@pytest.mark.requires_chapel_module("CSVMsg")
def test_pd_index_ea_to_csv_roundtrip(index_io_tmp):
    idx = _make_pd_index_with_arkouda_ea()
    assert idx.ak.is_arkouda()
    _assert_is_arkouda_backed_index(idx)

    with tempfile.TemporaryDirectory(dir=index_io_tmp) as tmp:
        out = os.path.join(tmp, "idx.csv")
        idx.ak.to_csv(out)

        rd = _read_index_from_csv(out)
        _assert_roundtrip_equal(idx, rd)


@pytest.mark.requires_chapel_module("ParquetMsg")
def test_pd_index_ea_to_parquet_roundtrip(index_io_tmp):
    idx = _make_pd_index_with_arkouda_ea()
    _assert_is_arkouda_backed_index(idx)

    with tempfile.TemporaryDirectory(dir=index_io_tmp) as tmp:
        out = os.path.join(tmp, "idx_parquet")
        idx.ak.to_parquet(out)

        rd = _read_index_from_parquet(f"{out}*")
        _assert_roundtrip_equal(idx, rd)


@pytest.mark.requires_chapel_module("HDFMsg")
def test_pd_index_ea_to_hdf_roundtrip(index_io_tmp):
    idx = _make_pd_index_with_arkouda_ea()
    _assert_is_arkouda_backed_index(idx)

    with tempfile.TemporaryDirectory(dir=index_io_tmp) as tmp:
        out = os.path.join(tmp, "idx_hdf")
        idx.ak.to_hdf(out)

        rd = _read_index_from_hdf(f"{out}*")
        _assert_roundtrip_equal(idx, rd)
