from logging import DEBUG, INFO, WARN, FileHandler, StreamHandler
import tempfile

import pytest

import arkouda as ak
from arkouda import logger
from arkouda.logger import LogLevel, getArkoudaClientLogger, getArkoudaLogger
from arkouda.pandas import io_util


class TestLogger:
    logger_test_base_tmp = f"{pytest.temp_directory}/logger_io_test"
    io_util.get_directory(logger_test_base_tmp)

    def test_logger_docstrings(self):
        import doctest

        result = doctest.testmod(logger, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_log_level(self):
        assert "DEBUG" == LogLevel.DEBUG.value
        assert "INFO" == LogLevel.INFO.value
        assert "WARN" == LogLevel.WARN.value
        assert "CRITICAL" == LogLevel.CRITICAL.value
        assert "ERROR" == LogLevel.ERROR.value

        assert LogLevel.DEBUG == LogLevel("DEBUG")
        assert LogLevel.INFO == LogLevel("INFO")
        assert LogLevel.WARN == LogLevel("WARN")
        assert LogLevel.CRITICAL == LogLevel("CRITICAL")
        assert LogLevel.ERROR == LogLevel("ERROR")

    def test_arkouda_logger(self):
        handler = StreamHandler()
        handler.name = "streaming"
        logger = getArkoudaLogger(name=self.__class__.__name__, handlers=[handler])
        assert DEBUG == logger.level
        assert "TestLogger" == logger.name
        assert logger.getHandler("streaming") is not None
        logger.debug("debug message")

    def test_arkouda_client_logger(self):
        logger = getArkoudaClientLogger(name="ClientLogger")
        assert DEBUG == logger.level
        assert "ClientLogger" == logger.name
        logger.debug("debug message")
        assert logger.getHandler("console-handler") is not None
        with pytest.raises(ValueError):
            logger.getHandler("console-handlers")

    def test_update_arkouda_logger_log_level(self):
        logger = getArkoudaLogger(name="UpdateLogger")
        assert DEBUG == logger.level
        logger.debug("debug before level change")
        logger.changeLogLevel(LogLevel.WARN)
        assert WARN == logger.handlers[0].level
        logger.debug("debug after level change")

        with tempfile.TemporaryDirectory(dir=TestLogger.logger_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/logger_test_output.txt"

            handler_one = StreamHandler()
            handler_one.name = "handler-one"
            handler_one.setLevel(DEBUG)
            handler_two = FileHandler(filename=file_name)
            handler_two.name = "handler-two"
            handler_two.setLevel(INFO)
            logger = getArkoudaLogger(name="UpdateLogger", handlers=[handler_one, handler_two])
            logger.changeLogLevel(level=LogLevel.WARN, handlerNames=["handler-one"])
            assert WARN == handler_one.level
            assert INFO == handler_two.level

    def test_verbosity_controls(self):
        logger = getArkoudaLogger(name="VerboseLogger", logLevel=LogLevel("INFO"))

        assert INFO == logger.getHandler("console-handler").level
        logger.debug("non-working debug message")
        logger.enableVerbose()
        assert DEBUG == logger.getHandler("console-handler").level
        logger.debug("working debug message")
        logger.disableVerbose()
        assert INFO == logger.getHandler("console-handler").level
        logger.debug("next non-working debug message")

    def test_enable_disable_verbose(self):
        logger_one = getArkoudaLogger(name="logger_one", logLevel=LogLevel.INFO)
        logger_two = getArkoudaLogger(name="logger_two", logLevel=LogLevel.INFO)

        logger_one.debug("logger_one before enableVerbose")
        logger_two.debug("logger_two before enableVerbose")
        ak.enableVerbose()
        logger_one.debug("logger_one after enableVerbose")
        logger_two.debug("logger_two after enableVerbose")
        ak.disableVerbose()
        logger_one.debug("logger_one after disableVerbose")
        logger_two.debug("logger_two after disableVerbose")

    def test_error_handling(self):
        logger = getArkoudaLogger(name="VerboseLogger", logLevel=LogLevel("INFO"))
        with pytest.raises(ValueError):
            logger.getHandler("not-a-handler")

        with pytest.raises(TypeError):
            logger.disableVerbose(logLevel="INFO")
