import unittest
from context import arkouda
from arkouda.logger import *
from logging import StreamHandler, DEBUG, INFO, WARN, \
     ERROR, CRITICAL, FileHandler
'''
Tests Arkouda logging functionality
'''
class LoggerTest(unittest.TestCase):
    
    def testLogLevel(self):
        self.assertEqual('DEBUG', LogLevel.DEBUG.value) 
        self.assertEqual('INFO', LogLevel.INFO.value) 
        self.assertEqual('WARN', LogLevel.WARN.value)  
        self.assertEqual('CRITICAL', LogLevel.CRITICAL.value)
        self.assertEqual('ERROR', LogLevel.ERROR.value)        
        
    def testArkoudaLogger(self): 
        handler = StreamHandler()
        handler.name = 'streaming'   
        logger = getArkoudaLogger(name=self.__class__.__name__, 
                                  handlers=[handler])
        self.assertEqual(DEBUG, logger.level)
        self.assertEqual('LoggerTest', logger.name)
        self.assertIsNotNone(logger.getHandler('streaming'))
        logger.debug('debug message')
        
    def testArkoudaClientLogger(self):
        logger = getArkoudaClientLogger(name='ClientLogger')
        self.assertEqual(DEBUG, logger.level)
        self.assertEqual('ClientLogger', logger.name)
        logger.debug('debug message')      
        self.assertIsNotNone(logger.getHandler('console-handler'))
        with self.assertRaises(ValueError):
            logger.getHandler('console-handlers')
        
    def testUpdateArkoudaLoggerLogLevel(self):  
        logger = getArkoudaLogger(name='UpdateLogger')
        self.assertEqual(DEBUG, logger.level)
        logger.debug('debug before level change')
        logger.changeLogLevel(LogLevel.WARN)
        self.assertEqual(WARN, logger.handlers[0].level)
        logger.debug('debug after level change')
        
        handlerOne = StreamHandler()
        handlerOne.name = 'handler-one'
        handlerOne.setLevel(DEBUG)
        handlerTwo = FileHandler(filename='/tmp/output.txt')
        handlerTwo.name = 'handler-two'
        handlerTwo.setLevel(INFO)
        logger = getArkoudaLogger(name='UpdateLogger', 
                                  handlers=[handlerOne,handlerTwo])
        logger.changeLogLevel(level=LogLevel.WARN, handlerNames=['handler-one'])
        self.assertEqual(WARN,handlerOne.level)
        self.assertEqual(INFO, handlerTwo.level)

    def testVerbosityControls(self):
        logger = ArkoudaLogger(name='VerboseLogger', logLevel=LogLevel('INFO'))

        self.assertEqual(INFO, logger.getHandler('console-handler').level)
        logger.debug('non-working debug message')     
        logger.enableVerbose()  
        self.assertEqual(DEBUG, logger.getHandler('console-handler').level)
        logger.debug('working debug message')     
        logger.disableVerbose() 
        self.assertEqual(INFO, logger.getHandler('console-handler').level)
        logger.debug('next non-working debug message') 