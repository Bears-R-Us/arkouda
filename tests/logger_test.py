import tempfile

from logging import DEBUG, INFO, WARN, FileHandler, StreamHandler

import pytest

import arkouda as ak

from arkouda import logger
from arkouda.logger import LogLevel, get_arkouda_client_logger, get_arkouda_logger
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
        logger = get_arkouda_logger(name=self.__class__.__name__, handlers=[handler])
        assert DEBUG == logger.level
        assert "TestLogger" == logger.name
        assert logger.get_handler("streaming") is not None
        logger.debug("debug message")

    def test_arkouda_client_logger(self):
        logger = get_arkouda_client_logger(name="ClientLogger")
        assert DEBUG == logger.level
        assert "ClientLogger" == logger.name
        logger.debug("debug message")
        assert logger.get_handler("console-handler") is not None
        with pytest.raises(ValueError):
            logger.get_handler("console-handlers")

    def test_update_arkouda_logger_log_level(self):
        logger = get_arkouda_logger(name="UpdateLogger")
        assert DEBUG == logger.level
        logger.debug("debug before level change")
        logger.change_log_level(LogLevel.WARN)
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
            logger = get_arkouda_logger(name="UpdateLogger", handlers=[handler_one, handler_two])
            logger.change_log_level(level=LogLevel.WARN, handlerNames=["handler-one"])
            assert WARN == handler_one.level
            assert INFO == handler_two.level

    def test_verbosity_controls(self):
        logger = get_arkouda_logger(name="VerboseLogger", log_level=LogLevel("INFO"))
        assert INFO == logger.get_handler("console-handler").level
        logger.debug("non-working debug message")
        logger.enable_verbose()
        assert DEBUG == logger.get_handler("console-handler").level
        logger.debug("working debug message")
        logger.disable_verbose()
        assert INFO == logger.get_handler("console-handler").level
        logger.debug("next non-working debug message")

    def test_enable_disable_verbose(self):
        logger_one = get_arkouda_logger(name="logger_one", log_level=LogLevel.INFO)
        logger_two = get_arkouda_logger(name="logger_two", log_level=LogLevel.INFO)
        logger_one.debug("logger_one before enable_verbose")
        logger_two.debug("logger_two before enable_verbose")
        ak.enable_verbose()
        logger_one.debug("logger_one after enable_verbose")
        logger_two.debug("logger_two after enable_verbose")
        ak.disable_verbose()
        logger_one.debug("logger_one after disable_verbose")
        logger_two.debug("logger_two after disable_verbose")

    def test_error_handling(self):
        logger = get_arkouda_logger(name="VerboseLogger", log_level=LogLevel("INFO"))
        with pytest.raises(ValueError):
            logger.get_handler("not-a-handler")

        with pytest.raises(TypeError):
            logger.disable_verbose(log_level="INFO")
