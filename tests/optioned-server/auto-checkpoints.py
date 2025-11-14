import os

from time import sleep

import pytest

import arkouda as ak

from arkouda.pandas.io_util import delete_directory, directory_exists


pytestmark = pytest.mark.skipif(
    os.environ.get("CHPL_HOST_PLATFORM") == "hpe-apollo",
    reason="skipped on CHPL_HOST_PLATFORM=hpe-apollo - login/compute file system access unreliable",
)

autockptPath = ".akdata"
autockptName = "auto_checkpoint"
autockptDir = f"{autockptPath}/{autockptName}"
autockptPrevName = f"{autockptName}.prev"
autockptPrevDir = f"{autockptPath}/{autockptPrevName}"

"""
When testing auto-checkpointing, we need to accommodate the unknown delays
before checkpointing starts. See directory_exists_delayed().

Another unknown delay is from when the checkpointing starts until it finishes,
especially it needs to save a noticeable amount of data. For that, we use
a testing helper that waits for an in-progres auto-checkpointing, if any,
to complete:
    ak.client.wait_for_async_activity()

`ak.load_checkpoint()` tells us whether an auto-checkpoint has been created
successfully. We do not test checkpoint correctness here. We leave that for
`checkpoint_test.py`, given that automatic and client-driven checkpoints
use the same implementation.
"""


def directory_exists_delayed(path, num_delays, delay=0.1):
    """
    Repeats directory_exists() query extra `num_delays` times,
    each after a delay of `delay` seconds.

    This allows testing to tolerate the delay from a completion
    of an idle-time period to the server actually starting checkpointing.
    This delay can be up to `checkpointCheckInterval`
    which defaults to min(`checkpointIdleTime`, `checkpointMemPctDelay`).
    """
    for _ in range(num_delays):
        if directory_exists(path):
            return True
        sleep(delay)
    return directory_exists(path)


@pytest.mark.skip_if_max_rank_greater_than(1)
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
        delete_directory(autockptPrevDir)
        a = ak.ones(pytest.prob_size[0])
        # expect auto-checkpointing after ~1 second [checkpointIdleTime]
        sleep(0.5)
        assert not directory_exists(autockptDir)
        sleep(1)
        assert directory_exists_delayed(autockptDir, 10)
        ak.client.wait_for_async_activity()
        delete_directory(autockptDir)
        b = ak.ones(pytest.prob_size[0])
        # expect auto-checkpointing after ~3 seconds [checkpointInterval]
        sleep(2)
        assert not directory_exists(autockptDir)
        sleep(1)
        assert directory_exists_delayed(autockptDir, 10)
        ak.client.wait_for_async_activity()
        c = ak.ones(pytest.prob_size[0])
        # expect another auto-checkpointing after ~3 seconds [checkpointInterval]
        # that will move autockptDir to autockptPrevDir and create a new autockptDir;
        # meanwhile, after 1 second this will not have happened yet:
        sleep(1)
        assert directory_exists(autockptDir)
        assert not directory_exists(autockptPrevDir)
        sleep(1)
        assert directory_exists_delayed(autockptPrevDir, 10)
        ak.client.wait_for_async_activity()
        assert directory_exists(autockptDir)
        # verify that these checkpoints were created successfully
        ak.load_checkpoint(autockptName)
        ak.load_checkpoint(autockptPrevName)
        delete_directory(autockptDir)
        delete_directory(autockptPrevDir)
        del a, b, c  # avoid flake8 errors about unused a,b,c


@pytest.mark.skip_if_max_rank_greater_than(1)
class TestMemPct:
    class_server_args = [
        "--checkpointMemPct=5",
        "--checkpointMemPctDelay=1",
        "--checkpointInterval=1",
    ]

    def test_memory_percentage(self):
        """
        Check the following sequence of events:
          allocate small memory
          2.5 [more than checkpointMemPctDelay since activity]
          verify no auto-checkpoint
          allocate "big" memory, i.e., over checkpointMemPct
          1.5 [more than checkpointMemPctDelay since activity]
          verify an auto-checkpoint exists
          2.5 [more than checkpointInterval since last checkpoint]
          verify no new auto-checkpoint, since no server activity since last A-CP
          perform server activity
          1.5 [more than checkpointMemPctDelay since activity]
          verify a new auto-checkpoint exists

        While class_server_args sets checkpointMemPctDelay and checkpointInterval
        to 1 second, we pad it to accomodate the following scenario:
        First, directory_exists_delayed() allows the CP daemon to wake up
        the first time in just under 1 second after server activity, realize that
        the waiting time of 1 second has not passed, then sleep for another
        1 second before taking action. We add 0.5 seconds on top of that
        to ensure the server completes whatever action it takes.

        """
        delete_directory(autockptDir)  # start with a clean slate
        avail_mem = ak.get_mem_avail()
        small_array = ak.zeros(100)  # below memory threshold
        sleep(2.5)
        assert not directory_exists(autockptDir)
        big_array = ak.zeros(int(avail_mem / 140))  # over 5% of avail_mem
        sleep(1.5)
        assert directory_exists_delayed(autockptDir, 10)
        ak.client.wait_for_async_activity()
        delete_directory(autockptDir)
        sleep(2.5)
        assert not directory_exists(autockptDir)
        small_2 = ak.zeros(100)  # some other server activity
        sleep(1.5)
        assert directory_exists_delayed(autockptDir, 10)
        # wait() is not needed since it is implicit in load_checkpoint()
        ak.load_checkpoint(autockptName)
        delete_directory(autockptDir)
        del small_array, small_2, big_array  # avoid flake8 errors about unused vars
