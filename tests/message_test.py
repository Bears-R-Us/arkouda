import unittest, json
from context import arkouda
from arkouda.message import RequestMessage, MessageFormat, ReplyMessage, \
     MessageType

class MessageTest(unittest.TestCase):

    def testMessageFormat(self):
        self.assertEqual(MessageFormat.BINARY, MessageFormat('BINARY'))
        self.assertEqual(MessageFormat.STRING, MessageFormat('STRING'))
        self.assertEqual('BINARY', str(MessageFormat.BINARY))
        self.assertEqual('STRING', str(MessageFormat.STRING))
        self.assertEqual('BINARY', repr(MessageFormat.BINARY))
        self.assertEqual('STRING', repr(MessageFormat.STRING))
        
        with self.assertRaises(ValueError):
            MessageFormat('STR')        
        
    def testMessageType(self):
        self.assertEqual(MessageType.NORMAL, MessageType('NORMAL'))
        self.assertEqual(MessageType.WARNING, MessageType('WARNING'))
        self.assertEqual(MessageType.ERROR, MessageType('ERROR'))
       
        self.assertEqual('NORMAL', str(MessageType.NORMAL))
        self.assertEqual('WARNING', str(MessageType.WARNING))
        self.assertEqual('ERROR', str(MessageType.ERROR))
        self.assertEqual('NORMAL', repr(MessageType.NORMAL))
        self.assertEqual('WARNING', repr(MessageType.WARNING))
        self.assertEqual('ERROR', repr(MessageType.ERROR))
        
        with self.assertRaises(ValueError):
            MessageType('STANDARD')
      
    def testRequestMessage(self):
        msg = RequestMessage(user='user1', token='token', cmd='connect', 
                      format=MessageFormat.STRING)
        msgDupe = RequestMessage(user='user1', token='token', cmd='connect', 
                      format=MessageFormat.STRING)
        msgNonDupe = RequestMessage(user='user1', token='token', cmd='connect', 
                      format=MessageFormat.BINARY)
        
        self.assertEqual('user1', msg.user)
        self.assertEqual('token', msg.token)
        self.assertEqual('connect', msg.cmd)
        self.assertEqual(MessageFormat.STRING, msg.format)
        
        self.assertEqual(msg,msgDupe)
        self.assertNotEqual(msg, msgNonDupe)
        
        self.assertEqual("RequestMessage(user='user1', token='token', cmd='connect', format=STRING, args=None)", 
                         str(msg))
        
        self.assertEqual("RequestMessage(user='user1', token='token', cmd='connect', format=STRING, args=None)", 
                         repr(msg))

        self.assertEqual('{"user": "user1", "token": "token", "cmd": "connect", "format": "STRING", "args": ""}',
                        json.dumps(msg.asdict()))
        
        self.assertFalse(self.assertRaises(Exception,json.loads(json.dumps(msg.asdict()))))
        
        minMsg = RequestMessage(user='user1', cmd='connect')
        
        self.assertEqual("RequestMessage(user='user1', token=None, cmd='connect', format=STRING, args=None)", 
                         str(minMsg))
        
        self.assertEqual("RequestMessage(user='user1', token=None, cmd='connect', format=STRING, args=None)", 
                         repr(minMsg))
        self.assertEqual('{"user": "user1", "token": "", "cmd": "connect", "format": "STRING", "args": ""}',
                        json.dumps(minMsg.asdict()))
        
    def testReplyMessage(self):
        msg = ReplyMessage(msg='normal result',msgType=MessageType.NORMAL, user='user')
        msgDupe = ReplyMessage(msg='normal result',msgType=MessageType.NORMAL, user='user')
        msgNonDupe = ReplyMessage(msg='normal result 2',msgType=MessageType.NORMAL, user='user')
        
        self.assertEqual(msg,msgDupe)
        self.assertNotEqual(msg, msgNonDupe)
        
        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", str(msg))
        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", repr(msg))
        
        newMsg = ReplyMessage.fromdict({ 'msg' : 'normal result', 'msgType': 'NORMAL', 'user': 'user'})
        self.assertEqual(msg,newMsg)    
        
        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", str(newMsg))
        self.assertEqual("ReplyMessage(msg='normal result', msgType=NORMAL, user='user')", repr(newMsg))    
        
        with self.assertRaises(ValueError):
            ReplyMessage.fromdict({ 'msg' : 'normal result', 'msgType': 'NORMAL'})
    