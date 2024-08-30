import json
import unittest

from base_test import ArkoudaTest

from arkouda.client import _json_args_to_str
from arkouda.client_dtypes import Fields, ip_address
from arkouda.message import MessageFormat, MessageType, ReplyMessage, RequestMessage
from arkouda.pdarraycreation import arange, array
from arkouda.timeclass import date_range


class MessageTest(unittest.TestCase):
    def testMessageFormat(self):
        self.assertEqual(MessageFormat.BINARY, MessageFormat("BINARY"))
        self.assertEqual(MessageFormat.STRING, MessageFormat("STRING"))
        self.assertEqual("BINARY", str(MessageFormat.BINARY))
        self.assertEqual("STRING", str(MessageFormat.STRING))
        self.assertEqual("BINARY", repr(MessageFormat.BINARY))
        self.assertEqual("STRING", repr(MessageFormat.STRING))

        with self.assertRaises(ValueError):
            MessageFormat("STR")

    def testMessageType(self):
        self.assertEqual(MessageType.NORMAL, MessageType("NORMAL"))
        self.assertEqual(MessageType.WARNING, MessageType("WARNING"))
        self.assertEqual(MessageType.ERROR, MessageType("ERROR"))

        self.assertEqual("NORMAL", str(MessageType.NORMAL))
        self.assertEqual("WARNING", str(MessageType.WARNING))
        self.assertEqual("ERROR", str(MessageType.ERROR))
        self.assertEqual("NORMAL", repr(MessageType.NORMAL))
        self.assertEqual("WARNING", repr(MessageType.WARNING))
        self.assertEqual("ERROR", repr(MessageType.ERROR))

        with self.assertRaises(ValueError):
            MessageType("STANDARD")

    def testRequestMessage(self):
        msg = RequestMessage(user="user1", token="token", cmd="connect", format=MessageFormat.STRING)
        msgDupe = RequestMessage(user="user1", token="token", cmd="connect", format=MessageFormat.STRING)
        msgNonDupe = RequestMessage(
            user="user1", token="token", cmd="connect", format=MessageFormat.BINARY
        )

        self.assertEqual("user1", msg.user)
        self.assertEqual("token", msg.token)
        self.assertEqual("connect", msg.cmd)
        self.assertEqual(MessageFormat.STRING, msg.format)

        self.assertEqual(msg, msgDupe)
        self.assertNotEqual(msg, msgNonDupe)

        self.assertEqual(
            "RequestMessage(user='user1', token='token', cmd='connect', format=STRING, args=None, size=-1)",
            str(msg),
        )

        self.assertEqual(
            "RequestMessage(user='user1', token='token', cmd='connect', format=STRING, args=None, size=-1)",
            repr(msg),
        )

        self.assertEqual(
            '{"user": "user1", "token": "token", "cmd": "connect", "format": "STRING", "args": "", "size": -1}',
            json.dumps(msg.asdict()),
        )

        self.assertFalse(self.assertRaises(Exception, json.loads(json.dumps(msg.asdict()))))

        minMsg = RequestMessage(user="user1", cmd="connect")

        self.assertEqual(
            "RequestMessage(user='user1', token=None, cmd='connect', format=STRING, args=None, size=-1)",
            str(minMsg),
        )

        self.assertEqual(
            "RequestMessage(user='user1', token=None, cmd='connect', format=STRING, args=None, size=-1)",
            repr(minMsg),
        )
        self.assertEqual(
            '{"user": "user1", "token": "", "cmd": "connect", "format": "STRING", "args": "", "size": -1}',
            json.dumps(minMsg.asdict()),
        )

    def testReplyMessage(self):
        msg = ReplyMessage(msg="normal result", msgType=MessageType.NORMAL, user="user")
        msgDupe = ReplyMessage(msg="normal result", msgType=MessageType.NORMAL, user="user")
        msgNonDupe = ReplyMessage(msg="normal result 2", msgType=MessageType.NORMAL, user="user")

        self.assertEqual(msg, msgDupe)
        self.assertNotEqual(msg, msgNonDupe)

        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", str(msg))
        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", repr(msg))

        newMsg = ReplyMessage.fromdict({"msg": "normal result", "msgType": "NORMAL", "user": "user"})
        self.assertEqual(msg, newMsg)

        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", str(newMsg))
        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", repr(newMsg))

        with self.assertRaises(ValueError):
            ReplyMessage.fromdict({"msg": "normal result", "msgType": "NORMAL"})


class JSONArgs(ArkoudaTest):
    def testJSONArgs(self):
        # test single value args
        size, args = _json_args_to_str({"arg1": "Test", "arg2": 5})
        self.assertEqual(size, 2)
        self.assertListEqual(
            [
                '{"key": "arg1", "dtype": "str", "val": "Test"}',
                '{"key": "arg2", "dtype": "int64", "val": "5"}',
            ],
            json.loads(args),
        )

        # test list arg of numerics
        size, args = _json_args_to_str({"list1": [3, 2, 4]})
        self.assertEqual(size, 1)
        self.assertListEqual(
            ['{"key": "list1", "dtype": "int64", "val": "[\\"3\\", \\"2\\", \\"4\\"]"}'],
            json.loads(args),
        )

        # test list of str
        size, args = _json_args_to_str({"list1": ["a", "b", "c"], "list2": ["d", "e", "f"]})
        self.assertEqual(size, 2)
        self.assertListEqual(
            [
                '{"key": "list1", "dtype": "str", "val": "[\\"a\\", \\"b\\", \\"c\\"]"}',
                '{"key": "list2", "dtype": "str", "val": "[\\"d\\", \\"e\\", \\"f\\"]"}',
            ],
            json.loads(args),
        )

        # test Datetime arg
        dt = date_range(start="2021-01-01 12:00:00", periods=100, freq="s")
        size, args = _json_args_to_str({"datetime": dt})
        self.assertEqual(size, 1)
        msgArgs = json.loads(json.loads(args)[0])
        self.assertEqual(msgArgs["key"], "datetime")
        self.assertEqual(msgArgs["dtype"], "int64")
        self.assertRegex(msgArgs["val"], "^id_\\w{7}_\\d+$")

        a = arange(10)
        ip = ip_address(a)
        size, args = _json_args_to_str({"ip": ip})
        self.assertEqual(size, 1)
        msgArgs = json.loads(json.loads(args)[0])
        self.assertEqual(msgArgs["key"], "ip")
        self.assertEqual(msgArgs["dtype"], "uint64")
        self.assertRegex(msgArgs["val"], "^id_\\w{7}_\\d+$")

        f = Fields(a, names="ABCD")
        size, args = _json_args_to_str({"fields": f})
        self.assertEqual(size, 1)
        msgArgs = json.loads(json.loads(args)[0])
        self.assertEqual(msgArgs["key"], "fields")
        self.assertEqual(msgArgs["dtype"], "uint64")
        self.assertRegex(msgArgs["val"], "^id_\\w{7}_\\d+$")

        # test list of pdarray
        pd1 = arange(3)
        pd2 = arange(4)
        size, args = _json_args_to_str({"pd1": pd1, "pd2": pd2})
        self.assertEqual(size, 2)
        for a in json.loads(args):
            p = json.loads(a)
            self.assertRegex(p["key"], "^pd(1|2)$")
            self.assertEqual(p["dtype"], "int64")
            self.assertRegex(p["val"], "^id_\\w{7}_\\d+$")

        # test list of Strings
        str1 = array(["abc", "def"])
        str2 = array(["Test", "Test2"])
        size, args = _json_args_to_str({"str1": str1, "str2": str2})
        self.assertEqual(size, 2)
        for a in json.loads(args):
            p = json.loads(a)
            self.assertRegex(p["key"], "^str(1|2)$")
            self.assertEqual(p["dtype"], "str")
            self.assertRegex(p["val"], "^id_\\w{7}_\\d+$")

        # test nested json
        size, args = _json_args_to_str(
            {
                "json_1": {
                    "param1": 1,
                    "param2": "abc",
                    "param3": [1, 2, 3],
                }
            }
        )
        self.assertEqual(size, 1)
        j = json.loads(json.loads(args)[0])
        self.assertEqual(j["key"], "json_1")
        self.assertEqual(j["dtype"], "dict")
        for i, a in enumerate(json.loads(j["val"])):
            p = json.loads(a)
            self.assertEqual(p["key"], f"param{i+1}")
            if i == 0:
                self.assertEqual(p["dtype"], "int64")
                self.assertEqual(p["val"], "1")
            elif i == 1:
                self.assertEqual(p["dtype"], "str")
                self.assertEqual(p["val"], "abc")
            else:
                self.assertEqual(p["dtype"], "int64")
                self.assertListEqual(json.loads(p["val"]), ["1", "2", "3"])
