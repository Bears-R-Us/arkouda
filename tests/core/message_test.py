import json

import pytest

import arkouda as ak

from arkouda.core import message
from arkouda.core.client import _json_args_to_str
from arkouda.core.message import MessageFormat, MessageType, ReplyMessage, RequestMessage


class TestMessage:
    def test_message_docstrings(self):
        import doctest

        result = doctest.testmod(message, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_message_format(self):
        assert MessageFormat.BINARY == MessageFormat("BINARY")
        assert MessageFormat.STRING == MessageFormat("STRING")
        assert "BINARY" == str(MessageFormat.BINARY)
        assert "STRING" == str(MessageFormat.STRING)
        assert "BINARY" == repr(MessageFormat.BINARY)
        assert "STRING" == repr(MessageFormat.STRING)

        with pytest.raises(ValueError):
            MessageFormat("STR")

    def test_message_type(self):
        assert MessageType.NORMAL == MessageType("NORMAL")
        assert MessageType.WARNING == MessageType("WARNING")
        assert MessageType.ERROR == MessageType("ERROR")

        assert "NORMAL" == str(MessageType.NORMAL)
        assert "WARNING" == str(MessageType.WARNING)
        assert "ERROR" == str(MessageType.ERROR)
        assert "NORMAL" == repr(MessageType.NORMAL)
        assert "WARNING" == repr(MessageType.WARNING)
        assert "ERROR" == repr(MessageType.ERROR)

        with pytest.raises(ValueError):
            MessageType("STANDARD")

    def test_request_msg(self):
        msg = RequestMessage(user="user1", token="token", cmd="connect", format=MessageFormat.STRING)
        msgDupe = RequestMessage(user="user1", token="token", cmd="connect", format=MessageFormat.STRING)
        msgNonDupe = RequestMessage(
            user="user1", token="token", cmd="connect", format=MessageFormat.BINARY
        )
        min_msg = RequestMessage(user="user1", cmd="connect")

        assert "user1" == msg.user
        assert "token" == msg.token
        assert "connect" == msg.cmd
        assert MessageFormat.STRING == msg.format

        assert msg == msgDupe
        assert msg != msgNonDupe

        rep_msg = (
            "RequestMessage(user='user1', token={}, cmd='connect', format=STRING, args=None, size=-1)"
        )
        assert rep_msg.format("'token'") == str(msg)
        assert rep_msg.format("'token'") == repr(msg)
        assert rep_msg.format("None") == str(min_msg)
        assert rep_msg.format("None") == repr(min_msg)

        dict_msg = (
            '{{"user": "user1", "token": {}, "cmd": "connect", "format": "STRING", "args": "",'
            ' "size": -1}}'
        )
        assert dict_msg.format('"token"') == json.dumps(msg.asdict())
        assert dict_msg.format('""') == json.dumps(min_msg.asdict())
        assert json.loads(json.dumps(msg.asdict())) == msg.asdict()

    def test_reply_msg(self):
        msg = ReplyMessage(msg="normal result", msgType=MessageType.NORMAL, user="user")
        msgDupe = ReplyMessage(msg="normal result", msgType=MessageType.NORMAL, user="user")
        msgNonDupe = ReplyMessage(msg="normal result 2", msgType=MessageType.NORMAL, user="user")

        assert msg == msgDupe
        assert msg != msgNonDupe

        assert "ReplyMessage(msg='normal result', msgType=NORMAL, user='user')" == str(msg)
        assert "ReplyMessage(msg='normal result', msgType=NORMAL, user='user')" == repr(msg)

        newMsg = ReplyMessage.fromdict({"msg": "normal result", "msgType": "NORMAL", "user": "user"})
        assert msg == newMsg

        assert "ReplyMessage(msg='normal result', msgType=NORMAL, user='user')" == str(newMsg)
        assert "ReplyMessage(msg='normal result', msgType=NORMAL, user='user')" == repr(newMsg)

        with pytest.raises(ValueError):
            ReplyMessage.fromdict({"msg": "normal result", "msgType": "NORMAL"})


class TestJSONArgs:
    # TODO numpy dtypes are not supported by json, we probably want to add an issue to handle this
    SCALAR_TYPES = [int, float, bool, str]
    #   The types below are support in arkouda, as noted in registration-config.json.  This may be
    #   the same issue noted in the above comment.
    SUPPORTED_TYPES = [ak.bool_, ak.uint64, ak.int64, ak.bigint, ak.uint8, ak.float64]

    @pytest.mark.parametrize("dtype", SCALAR_TYPES)
    def test_scalar_args(self, dtype):
        val1 = dtype(5)
        val2 = dtype(0)
        size, args = _json_args_to_str({"arg1": val1, "arg2": val2})
        expected = json.dumps(
            [
                json.dumps(
                    {
                        "key": "arg1",
                        "dtype": ak.resolve_scalar_dtype(val1),
                        "val": str(val1),
                    }
                ),
                json.dumps(
                    {
                        "key": "arg2",
                        "dtype": ak.resolve_scalar_dtype(val2),
                        "val": str(val2),
                    }
                ),
            ]
        )
        assert args == expected

    def test_addl_str(self):
        val = "abc"
        size, args = _json_args_to_str({"arg": val})
        expected = json.dumps(
            [
                json.dumps({"key": "arg", "dtype": ak.resolve_scalar_dtype(val), "val": val}),
            ]
        )
        assert args == expected

    @pytest.mark.parametrize("dtype", SCALAR_TYPES)
    def test_list_arg(self, dtype):
        l1 = [dtype(x) for x in [0, 1, 2, 3]]
        l2 = [dtype(x) for x in [9, 8, 7]]

        size, args = _json_args_to_str({"list1": l1, "list2": l2})
        expected = json.dumps(
            [
                json.dumps(
                    {
                        "key": "list1",
                        "dtype": ak.resolve_scalar_dtype(l1[0]),
                        "val": json.dumps([str(x) for x in l1]),
                    }
                ),
                json.dumps(
                    {
                        "key": "list2",
                        "dtype": ak.resolve_scalar_dtype(l2[0]),
                        "val": json.dumps([str(x) for x in l2]),
                    }
                ),
            ]
        )
        assert args == expected

    def test_list_addl_str(self):
        string_list = ["abc", "def", "l", "mn", "op"]
        size, args = _json_args_to_str({"str_list": string_list})

        expected = json.dumps(
            [
                json.dumps(
                    {
                        "key": "str_list",
                        "dtype": ak.resolve_scalar_dtype(string_list[0]),
                        "val": json.dumps(string_list),
                    }
                ),
            ]
        )
        assert args == expected

    def test_datetime_arg(self):
        dt = ak.date_range(start="2021-01-01 12:00:00", periods=100, freq="s")
        size, args = _json_args_to_str({"datetime": dt})

        expected = json.dumps([json.dumps({"key": "datetime", "dtype": "int64", "val": dt.name})])
        assert args == expected

    def test_ip_arg(self):
        a = ak.arange(10)
        ip = ak.ip_address(a)
        size, args = _json_args_to_str({"ip": ip})
        expected = json.dumps([json.dumps({"key": "ip", "dtype": "uint64", "val": ip.name})])
        assert args == expected

    def test_fields_arg(self):
        a = ak.arange(10)
        f = ak.Fields(a, names="ABCD")
        size, args = _json_args_to_str({"fields": f})
        expected = json.dumps([json.dumps({"key": "fields", "dtype": "uint64", "val": f.name})])
        assert args == expected

    @pytest.mark.parametrize("dtype", SUPPORTED_TYPES)
    def test_pda_arg(self, dtype):
        pda1 = ak.arange(3, dtype=dtype)
        pda2 = ak.arange(4, dtype=dtype)
        size, args = _json_args_to_str({"pda1": pda1, "pda2": pda2})
        expected = json.dumps(
            [
                json.dumps({"key": "pda1", "dtype": str(pda1.dtype), "val": pda1.name}),
                json.dumps({"key": "pda2", "dtype": str(pda2.dtype), "val": pda2.name}),
            ]
        )
        assert args == expected

        size, args = _json_args_to_str({"pda_list": [pda1, pda2]})
        expected = json.dumps(
            [
                json.dumps(
                    {
                        "key": "pda_list",
                        "dtype": ak.pdarray.objType,
                        "val": json.dumps([pda1.name, pda2.name]),
                    }
                ),
            ]
        )
        assert args == expected

    def test_segstr_arg(self):
        str1 = ak.array(["abc", "def"])
        str2 = ak.array(["Test", "Test2"])
        size, args = _json_args_to_str({"str1": str1, "str2": str2})
        expected = json.dumps(
            [
                json.dumps({"key": "str1", "dtype": "str", "val": str1.name}),
                json.dumps({"key": "str2", "dtype": "str", "val": str2.name}),
            ]
        )
        assert args == expected

        size, args = _json_args_to_str({"str_list": [str1, str2]})
        expected = json.dumps(
            [
                json.dumps(
                    {
                        "key": "str_list",
                        "dtype": ak.Strings.objType,
                        "val": json.dumps([str1.name, str2.name]),
                    }
                ),
            ]
        )
        assert args == expected

    def test_dict_arg(self):
        json_1 = {
            "param1": 1,
            "param2": "abc",
            "param3": [1, 2, 3],
            "param4": ak.arange(10),
            "param5": ak.array(["abc", "123"]),
        }
        data = {"json_1": json_1}
        size, args = _json_args_to_str(data)
        expected = json.dumps(
            [
                json.dumps(
                    {
                        "key": "json_1",
                        "dtype": "dict",
                        "val": json.dumps(
                            [
                                json.dumps(
                                    {
                                        "key": "param1",
                                        "dtype": ak.resolve_scalar_dtype(json_1["param1"]),
                                        "val": "1",
                                    }
                                ),
                                json.dumps(
                                    {
                                        "key": "param2",
                                        "dtype": ak.resolve_scalar_dtype(json_1["param2"]),
                                        "val": "abc",
                                    }
                                ),
                                json.dumps(
                                    {
                                        "key": "param3",
                                        "dtype": ak.resolve_scalar_dtype(json_1["param3"][0]),
                                        "val": json.dumps([str(x) for x in json_1["param3"]]),
                                    }
                                ),
                                json.dumps(
                                    {
                                        "key": "param4",
                                        "dtype": str(json_1["param4"].dtype),
                                        "val": json_1["param4"].name,
                                    }
                                ),
                                json.dumps(
                                    {
                                        "key": "param5",
                                        "dtype": "str",
                                        "val": json_1["param5"].name,
                                    }
                                ),
                            ]
                        ),
                    }
                )
            ]
        )
        assert args == expected
