from time import sleep

import pytest

import arkouda as ak
from arkouda.io_util import delete_directory, directory_exists

autockptPath = ".akdata"
autockptName = "auto_checkpoint"
autockptDir = f"{autockptPath}/{autockptName}"


def directory_exists_delayed(path, num_delays, delay=0.1):
    """
    Repeats directory_exists() checks num_delays times,
    each after a delay of `delay` seconds.
    This allows us to adjust for the server that can delay
    the detection of a condition by --checkpointCheckInterval, if set,
    otherwise by min(--checkpointIdleTime, --checkpointMemPctDelay).
    """
    for _ in range(num_delays):
        if directory_exists(path):
            return True
        sleep(delay)
    return directory_exists(path)


def usethem(*args):
    """
    dummy use to suppress flake8 complaints
    """
    pass


class TestIdleAndInterval:
    class_server_args = [
        "--checkpointIdleTime=1",
        "--checkpointInterval=3",
    ]

    def test_idletime_and_interval(self):
        """
        Check the following sequence of events, numbers indicating wait seconds:
          perform server activity
          .5 [less than checkpointIdleTime since activity]
          verify no auto-checkpoint
          1. [more than checkpointIdleTime since activity]
          veryfy auto-checkpoint exists
          another server activity
          2. [more than checkpointIdleTime since latest activity,
              less than checkpointInterval since latest checkpoint]
          verify no new auto-checkpoint
          1. [more than checkpointIdleTime since latest activity,
              more than checkpointInterval since latest checkpoint]
          verify new auto-checkpoint exists
        """

        delete_directory(autockptDir)  # start with a clean slate
        a = ak.ones(pytest.prob_size[0])
        sleep(.5)
        assert not directory_exists(autockptDir)
        sleep(1)
        assert directory_exists_delayed(autockptDir, 10)
        delete_directory(autockptDir)
        b = ak.ones(pytest.prob_size[0])
        sleep(2)
        assert not directory_exists(autockptDir)
        sleep(1)
        assert directory_exists_delayed(autockptDir, 10)
        delete_directory(autockptDir)
        usethem(a,b)
