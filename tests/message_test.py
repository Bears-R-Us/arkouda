import unittest, json
import dataclasses
from context import arkouda
from arkouda.message import Message, MessageFormat

class MessageTest(unittest.TestCase):

    def testMessageFormat(self):
        self.assertEqual(MessageFormat.BINARY, MessageFormat('BINARY'))
        self.assertEqual(MessageFormat.STRING, MessageFormat('STRING'))
        self.assertEqual('BINARY', str(MessageFormat.BINARY))
        self.assertEqual('STRING', str(MessageFormat.STRING))
        self.assertEqual('BINARY', repr(MessageFormat.BINARY))
        self.assertEqual('STRING', repr(MessageFormat.STRING))
        
    def testMessage(self):
        msg = Message(user='user1', token='token', cmd='connect', 
                      format=MessageFormat.STRING)
        msgDupe = Message(user='user1', token='token', cmd='connect', 
                      format=MessageFormat.STRING)
        msgNonDupe = Message(user='user1', token='token', cmd='connect', 
                      format=MessageFormat.BINARY)
        
        self.assertEqual('user1', msg.user)
        self.assertEqual('token', msg.token)
        self.assertEqual('connect', msg.cmd)
        self.assertEqual(MessageFormat.STRING, msg.format)
        
        self.assertEqual(msg,msgDupe)
        self.assertNotEqual(msg, msgNonDupe)
        
        self.assertEqual("Message(user='user1', token='token', cmd='connect', format=STRING, args='')", 
                         str(msg))
        
        self.assertEqual("Message(user='user1', token='token', cmd='connect', format=STRING, args='')", 
                         repr(msg))

        self.assertEqual('{"user": "user1", "token": "token", "cmd": "connect", "format": "STRING", "args": ""}',
                        json.dumps(msg.asdict()))
        
        self.assertFalse(self.assertRaises(Exception,json.loads(json.dumps(msg.asdict()))))
        
        minimumMsg = Message(user='user1', cmd='connect')
        self.assertEqual('{"user": "user1", "token": "", "cmd": "connect", "format": "STRING", "args": ""}',
                        json.dumps(minimumMsg.asdict()))
        
        minimumMsg = Message(user='user1', cmd='connect')       