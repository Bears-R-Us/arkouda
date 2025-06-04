from time import sleep

import pytest

import arkouda as ak
from arkouda.io_util import delete_directory, directory_exists

autockptPath = ".akdata"
autockptName = "auto_checkpoint"
autockptDir = f"{autockptPath}/{autockptName}"
autockptPrev = f"{autockptPath}/{autockptName}.prev"


def directory_exists_delayed(path, num_delays, delay=0.1):
    """
    Repeats directory_exists() query num_delays times,
    each after a delay of `delay` seconds.

    This allows us to adjust for the server that can delay
    the detection of a condition by up to `checkpointCheckInterval`,
    which defaults to min(`checkpointIdleTime`, `checkpointMemPctDelay`).

    The final sleep() is needed to wait for server checkpointing to complete,
    once the directory is created. Otherwise delete_directory() executed
    by the tests may interfer with checkpointing.
    """
    for _ in range(num_delays):
        if directory_exists(path):
            sleep(0.3)  # conservative estimate
            return True
        sleep(delay)
    return False


class TestIdleAndInterval:
    class_server_args = [
        "--checkpointIdleTime=1",
        "--checkpointInterval=3",
    ]

    def test_idletime_and_interval(self):
        """
        Check the following sequence of events, numbers indicating wait seconds:
          perform server activity
          0.5 [less than checkpointIdleTime since activity]
          verify no auto-checkpoint
          1.0 [more than checkpointIdleTime since activity]
          veryfy an auto-checkpoint exists
          another server activity
          2.0 [more than checkpointIdleTime since latest activity,
               less than checkpointInterval since latest checkpoint]
          verify no new auto-checkpoint
          1.0 [more than checkpointIdleTime since latest activity,
               more than checkpointInterval since latest checkpoint]
          verify a new auto-checkpoint exists
          another server activity
          3.0 [more than checkpointInterval]
          verify existence of an auto-checkpoint and a previous auto-checkpoint
        """

        delete_directory(autockptDir)  # start with a clean slate
        delete_directory(autockptPrev)
        a = ak.ones(pytest.prob_size[0])
        # expect auto-checkpointing after ~1 second [checkpointIdleTime]
        sleep(0.5)
        assert not directory_exists(autockptDir)
        sleep(1)
        assert directory_exists_delayed(autockptDir, 10)
        delete_directory(autockptDir)
        b = ak.ones(pytest.prob_size[0])
        # expect auto-checkpointing after ~3 seconds [checkpointInterval]
        sleep(2)
        assert not directory_exists(autockptDir)
        sleep(1)
        assert directory_exists_delayed(autockptDir, 10)
        c = ak.ones(pytest.prob_size[0])
        # expect another auto-checkpointing after ~3 seconds [checkpointInterval]
        # that will move autockptDir to autockptPrev and create a new autockptDir
        sleep(1)
        assert directory_exists(autockptDir)
        assert not directory_exists(autockptPrev)
        sleep(1)
        assert directory_exists_delayed(autockptPrev, 10)
        assert directory_exists(autockptDir)
        delete_directory(autockptDir)
        delete_directory(autockptPrev)
        del a, b, c  # avoid flake8 errors about unused a,b,c


class TestMemPct:
    class_server_args = [
        "--checkpointMemPct=5",
        "--checkpointMemPctDelay=1",
        "--checkpointInterval=1",
    ]

    def test_memory_percentage(self):
        """
        Check the following sequence of events. While class_server_args sets
        checkpointMemPctDelay and checkpointInterval to 1 second, we pad it.
        First, directory_exists_delayed() allows the CP daemon to wake up
        the first time in just under 1 second after server activity, realize that
        the waiting time of 1 second has not passed, then sleep for another
        1 second before taking action. We add 0.5 seconds on top of that
        to ensure the server completes whatever action it takes.

        Here is the expected sequence:
          allocate small memory
          2.5 [more than checkpointMemPctDelay since activity]
          verify no auto-checkpoint
          allocate "big" memory, i.e., over checkpointMemPct
          2.5 [more than checkpointMemPctDelay since activity]
          verify an auto-checkpoint exists
          2.5 [more than checkpointInterval since last checkpoint]
          verify no new auto-checkpoint, since no server activity since last A-CP
          perform server activity
          2.5 [more than checkpointMemPctDelay since activity]
          verify a new auto-checkpoint exists
        """

        delete_directory(autockptDir)  # start with a clean slate
        avail_mem = ak.get_mem_avail()
        small_array = ak.zeros(100)  # below memory threshold
        sleep(2.5)
        assert not directory_exists(autockptDir)
        big_array = ak.zeros(int(avail_mem / 140))  # over 5% of avail_mem
        sleep(1.5)
        assert directory_exists_delayed(autockptDir, 10)
        delete_directory(autockptDir)
        sleep(2.5)
        assert not directory_exists(autockptDir)
        del small_array
        sleep(1.5)
        assert directory_exists_delayed(autockptDir, 10)
        delete_directory(autockptDir)
        del big_array  # avoid flake8 errors about unused big_array
