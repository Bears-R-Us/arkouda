import arkouda as ak
from arkouda.client import generic_msg

def test_command():
    rep_msg = generic_msg(cmd='test-command')

ak.__dict__["test_command"] = test_command
