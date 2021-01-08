import os, unittest
from context import arkouda
import arkouda as ak
from arkouda.logger import LogLevel, getArkoudaLogger, getArkoudaClientLogger
from logging import StreamHandler, DEBUG, INFO, WARN, \
     FileHandler

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
        
        self.assertEqual(LogLevel.DEBUG,LogLevel('DEBUG'))    
        self.assertEqual(LogLevel.INFO,LogLevel('INFO'))  
        self.assertEqual(LogLevel.WARN,LogLevel('WARN')) 
        self.assertEqual(LogLevel.CRITICAL,LogLevel('CRITICAL')) 
        self.assertEqual(LogLevel.ERROR,LogLevel('ERROR')) 
        
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
        handlerTwo = FileHandler(filename='output.txt')
        handlerTwo.name = 'handler-two'
        handlerTwo.setLevel(INFO)
        logger = getArkoudaLogger(name='UpdateLogger', 
                                  handlers=[handlerOne,handlerTwo])
        logger.changeLogLevel(level=LogLevel.WARN, handlerNames=['handler-one'])
        self.assertEqual(WARN,handlerOne.level)
        self.assertEqual(INFO, handlerTwo.level)

    def testVerbosityControls(self):
        logger = getArkoudaLogger(name='VerboseLogger', logLevel=LogLevel('INFO'))

        self.assertEqual(INFO, logger.getHandler('console-handler').level)
        logger.debug('non-working debug message')     
        logger.enableVerbose()  
        self.assertEqual(DEBUG, logger.getHandler('console-handler').level)
        logger.debug('working debug message')     
        logger.disableVerbose() 
        self.assertEqual(INFO, logger.getHandler('console-handler').level)
        logger.debug('next non-working debug message') 
        
    def testEnableDisableVerbose(self):  
        loggerOne = getArkoudaLogger(name='loggerOne',logLevel=LogLevel.INFO)
        loggerTwo = getArkoudaLogger('loggerTwo', logLevel=LogLevel.INFO)
        
        loggerOne.debug('loggerOne before enableVerbose')
        loggerTwo.debug('loggerTwo before enableVerbose')
        ak.enableVerbose()
        loggerOne.debug('loggerOne after enableVerbose')
        loggerTwo.debug('loggerTwo after enableVerbose')     
        ak.disableVerbose()
        loggerOne.debug('loggerOne after disableVerbose')
        loggerTwo.debug('loggerTwo after disableVerbose')  
        
    def testErrorHandling(self):
        logger = getArkoudaLogger(name='VerboseLogger', logLevel=LogLevel('INFO'))

        with self.assertRaises(ValueError):
            logger.getHandler('not-a-handler')
            
        with self.assertRaises(TypeError):
            logger.disableVerbose(logLevel='INFO')
    
    @classmethod
    def tearDownClass(cls):
        os.remove('output.txt')            