import unittest
from context import arkouda
from arkouda.logger import ArkoudaLogger, LogLevel


'''
Tests Arklouda logging functionality
'''
class LoggerTest(unittest.TestCase):
    
    def testLogLevel(self):
        self.assertEqual('DEBUG', LogLevel.DEBUG.value) 
        self.assertEqual('INFO', LogLevel.INFO.value) 
        self.assertEqual('WARN', LogLevel.WARN.value)  
        self.assertEqual('CRITICAL', LogLevel.CRITICAL.value)
        self.assertEqual('ERROR', LogLevel.ERROR.value)        
        
    def testArkoudaLogger(self):
        
        logger = ArkoudaLogger(name=self.__class__.__name__, 
                               level=LogLevel.DEBUG)
        self.assertEqual(LogLevel.DEBUG, logger.logLevel)
        self.assertEqual('LoggerTest', logger.name)
        logger.debug('debug message')
        
    def testArkoudaClientLogger(self):
        logger = ArkoudaLogger(name='ClientLogger', 
                               level=LogLevel.DEBUG, logFormat='')
        self.assertEqual(LogLevel.DEBUG, logger.logLevel)
        self.assertEqual('ClientLogger', logger.name)
        logger.debug('debug message')        
