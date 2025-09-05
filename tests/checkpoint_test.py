from datetime import datetime
import json
import os
from os import path
from shutil import rmtree
import tempfile

import pytest

import arkouda as ak
from arkouda.pandas import io_util


@pytest.fixture
def cp_test_base_tmp(request):
    # make sure to create a unique directory
    timestamp = str(datetime.now()).replace(" ", "_")
    cp_test_base_tmp = "{}/.cp_test_{}".format(os.getcwd(), timestamp)
    while path.isdir(cp_test_base_tmp):
        timestamp = str(datetime.now()).replace(" ", "_")
        cp_test_base_tmp = "{}/.cp_test_{}".format(os.getcwd(), timestamp)

    io_util.get_directory(cp_test_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(cp_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return cp_test_base_tmp


class TestCheckpoint:
    @pytest.mark.skip_if_max_rank_greater_than(1)
    @pytest.mark.skipif(
        os.environ.get("CHPL_HOST_PLATFORM") == "hpe-apollo",
        reason="skipped on CHPL_HOST_PLATFORM=hpe-apollo - login/compute file system access unreliable",
    )
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", ["int64", "float64", "bool"])
    def test_checkpoint(self, prob_size, dtype):
        rmtree(".akdata", ignore_errors=True)  # start from clean
        val2 = 2 if dtype != "bool" else True
        val3 = 3 if dtype != "bool" else False

        arr = ak.zeros(prob_size, dtype)
        arr[2] = val2

        cp_name = ak.save_checkpoint()

        expected_dir = ".akdata/{}".format(cp_name)

        try:
            # check basics
            assert path.isdir(expected_dir)
            assert path.isfile(path.join(expected_dir, "server.md"))

            arr[3] = val3

            # should overwrite the value
            ak.load_checkpoint(cp_name)

            assert arr[3] == 0
            assert arr[2] == val2
            assert arr.dtype == dtype

        finally:
            rmtree(expected_dir)

    @pytest.mark.skip_if_max_rank_greater_than(1)
    def test_bigint_checkpoint(self, cp_test_base_tmp):
        prob_size = 300  # force memBuf resizing in the implementation
        lst = list()
        pda = ak.zeros(prob_size, dtype=ak.bigint)

        cur = 37
        for i in range(prob_size - 1):
            cur *= 347
            pda[i] = cur
            lst.append(cur)

        pda[prob_size - 1] = 3
        lst.append(3)

        with tempfile.TemporaryDirectory(dir=cp_test_base_tmp) as tmp_dirname:
            cp_name = ak.save_checkpoint(path=tmp_dirname)
            for i in range(prob_size):
                pda[i] = i  # clobber the elements
            ak.load_checkpoint(path=tmp_dirname, name=cp_name)  # restore 'pda'

            for i in range(prob_size):
                assert pda[i] == lst[i]

    @pytest.mark.skip_if_max_rank_greater_than(1)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_checkpoint_custom_names(self, cp_test_base_tmp, prob_size):
        arr = ak.zeros(prob_size, int)
        arr[2] = 2

        cp_name = "test_cp"

        with tempfile.TemporaryDirectory(dir=cp_test_base_tmp) as tmp_dirname:
            ret = ak.save_checkpoint(path=tmp_dirname, name=cp_name)
            assert ret == cp_name

            # check basics
            assert path.isdir(tmp_dirname)

            expected_dir = path.join(tmp_dirname, cp_name)
            assert path.isdir(expected_dir)
            assert path.isfile(path.join(expected_dir, "server.md"))

            arr[3] = 3

            # should overwrite the value
            ak.load_checkpoint(path=tmp_dirname, name=cp_name)

            assert arr[3] == 0
            assert arr[2] == 2

    @pytest.mark.skip_if_max_rank_greater_than(1)
    def test_incorrect_nl(self):
        cp_name = "test_incorrect_nl_cp"
        create_fake_cp(cp_name, num_locales=ak.get_config()["numLocales"] + 1)
        try:
            ak.load_checkpoint(cp_name)
            raise AssertionError("Expected RuntimeError was not raised")
        except RuntimeError as err:
            assert (
                "Attempting to load a checkpoint that was made with a different number of locales"
            ) in str(err)
        finally:
            clean_fake_cp(cp_name)

    @pytest.mark.skip_if_max_rank_greater_than(1)
    def test_incorrect_chunks(self):
        cp_name = "test_incorrect_chunks_cp"
        create_fake_cp(cp_name)

        metadata_name = create_fake_array(cp_name)

        with open(metadata_name, "a") as f:
            for i in range(0, 0):  # do not generate any chunks to test an error
                f.write(json.dumps({"filename": "dummy file", "numElems": 100}))

        try:
            ak.load_checkpoint(cp_name)
            raise AssertionError("Expected RuntimeError was not raised")
        except RuntimeError as err:
            assert "could not read chunk 1 metadata" in str(err)
        finally:
            clean_fake_cp(cp_name)

    @pytest.mark.skip_if_max_rank_greater_than(1)
    def test_corrupt_json(self):
        cp_name = "test_corrupt_json_cp"
        create_fake_cp(cp_name)
        create_fake_array(cp_name, corrupt_json=True)

        try:
            ak.load_checkpoint(cp_name)
            raise AssertionError("Expected RuntimeError was not raised")
        except RuntimeError as err:
            assert "field 'size' not found" in str(err)
        finally:
            clean_fake_cp(cp_name)

    @pytest.mark.skip_if_max_rank_greater_than(1)
    def test_wrong_argument(self):
        try:
            ak.save_checkpoint(mode="override")
            raise AssertionError("Expected RuntimeError was not raised")
        except ValueError as err:
            assert 'invalid checkpointing mode "override"' in str(err)


def create_fake_array(cp_name, arr_name="dummy", num_target_locales=-1, corrupt_json=False):
    cp_path = get_def_cp_path(cp_name)

    arr_metadata = path.join(cp_path, "{}.md".format(arr_name))

    if num_target_locales == -1:
        num_target_locales = ak.get_config()["numLocales"]

    size_field = "size" if not corrupt_json else "junk"

    with open(arr_metadata, "w") as f:
        f.write(
            json.dumps(
                {
                    "entryName": arr_name,
                    "entryType": "PrimitiveTypedArraySymEntry",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "dtype": "int64",
                    size_field: 10,
                    "ndim": 1,
                    "numTargetLocales": num_target_locales,
                    "numChunks": num_target_locales,
                }
            )
            + "\n"
        )

    return arr_metadata


def create_fake_cp(cp_name, serverid="dummy", num_locales=-1):
    clean_fake_cp(cp_name)
    cp_path = get_def_cp_path(cp_name)

    os.makedirs(cp_path)

    if num_locales == -1:
        num_locales = ak.get_config()["numLocales"]

    server_metadata = path.join(cp_path, "server.md")

    # write a server metadata with mismatching number of locales
    with open(server_metadata, "w") as f:
        f.write(json.dumps({"serverid": "dummy", "numLocales": num_locales}))


def clean_fake_cp(cp_name):
    rmtree(get_def_cp_path(cp_name), ignore_errors=True)


def get_def_cp_path(cp_name):
    def_cp_root = ".akdata"
    return path.join(def_cp_root, cp_name)
