import arkouda as ak

from os import path
from shutil import rmtree
from datetime import datetime

class TestCheckpoint:
    def test_checkpoint(self):
        arr = ak.zeros(10, int)
        arr[2] = 2

        cp_name = ak.save_checkpoint()

        expected_dir = '.akdata/{}'.format(cp_name)

        try:
            # check basics
            assert(path.isdir(expected_dir))
            assert(path.isfile(path.join(expected_dir, 'server.md')))

            arr[3] = 3

            # should overwrite the value
            loaded = ak.load_checkpoint(cp_name)

            assert(arr[3] == 0)

        finally:
            rmtree(expected_dir)

    def test_checkpoint_custom_names(self):
        arr = ak.zeros(10, int)
        arr[2] = 2

        cp_name = 'test_cp'

        # make sure to create a unique directory
        expected_path = str(datetime.now()).replace(' ', '_')
        while path.isdir(expected_path):
            expected_path = str(datetime.now()).replace(' ', '_')

        try:
            ret = ak.save_checkpoint(path=expected_path, name=cp_name)
            assert(ret == cp_name)


            # check basics
            assert(path.isdir(expected_path))

            expected_dir = path.join(expected_path, cp_name)
            assert(path.isdir(expected_dir))
            assert(path.isfile(path.join(expected_dir, 'server.md')))

            arr[3] = 3

            # should overwrite the value
            ak.load_checkpoint(path=expected_path, name=cp_name)

            assert(arr[3] == 0)

        finally:
            rmtree(expected_path)

# def main():
    # ak.connect(connect_url='tcp://dev:5555')
    # cp_test = TestCheckpoint()

    # cp_test.test_checkpoint()

# if __name__ == '__main__':
    # main()
