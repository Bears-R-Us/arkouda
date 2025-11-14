import logging

from collections.abc import Iterator  # or: from typing import Iterator
from typing import Literal

import pytest

import arkouda as ak

from arkouda.numpy import err as akerr


class TestErr:
    def test_err_docstrings(self):
        import doctest

        from arkouda.numpy import err

        result = doctest.testmod(err, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.fixture(autouse=True)
    def restore_err_state(self) -> Iterator[None]:
        """
        Save & restore global error state and errcall around each test.

        Yields
        ------
        None
            This fixture yields control to the test, then restores Arkouda's
            error state and error callback afterwards.
        """
        old_modes = akerr.geterr()
        old_call = akerr.geterrcall()
        try:
            yield
        finally:
            akerr.seterr(**old_modes)
            akerr.seterrcall(old_call)

    # ---------- basic API & defaults ----------

    def test_defaults_are_ignore_for_all_categories(self):
        modes = akerr.geterr()
        assert modes == {"divide": "ignore", "over": "ignore", "under": "ignore", "invalid": "ignore"}

    def test_seterr_roundtrip_and_geterr(self):
        old = akerr.seterr(divide="warn")
        try:
            assert akerr.geterr()["divide"] == "warn"
        finally:
            akerr.seterr(**old)
        assert akerr.geterr()["divide"] == old["divide"]

    def test_seterr_rejects_unknown_key(self):
        with pytest.raises(ValueError):
            akerr.seterr(foo="warn")  # type: ignore[arg-type]

    def test_seterr_rejects_invalid_value(self):
        with pytest.raises(ValueError):
            akerr.seterr(divide="nope")  # type: ignore[arg-type]

    def test_seterr_warns_for_unimplemented_categories(self):
        # seterr() emits a warning for over/under/invalid (stubbed)
        with pytest.warns(UserWarning):
            akerr.seterr(over="warn")
        with pytest.warns(UserWarning):
            akerr.seterr(under="warn")
        with pytest.warns(UserWarning):
            akerr.seterr(invalid="warn")

    # ---------- seterrcall ----------

    def test_seterrcall_and_geterrcall_roundtrip(self):
        def handler(kind, msg):
            pass

        old = akerr.seterrcall(handler)
        try:
            assert akerr.geterrcall() is handler
        finally:
            akerr.seterrcall(old)

    def test_seterrcall_type_validation(self):
        with pytest.raises(TypeError):
            akerr.seterrcall(123)  # type: ignore[arg-type]

    # ---------- errstate context manager ----------

    def test_errstate_temporarily_sets_and_restores_modes(self):
        before = akerr.geterr()
        with akerr.errstate(divide="raise"):
            assert akerr.geterr()["divide"] == "raise"
        # restored
        after = akerr.geterr()
        assert after == before

    def test_errstate_temporarily_sets_and_restores_call(self):
        def temp_handler(kind, msg):
            pass

        before = akerr.geterrcall()
        with akerr.errstate(divide="call", call=temp_handler):
            assert akerr.geterrcall() is temp_handler
            assert akerr.geterr()["divide"] == "call"
        # restored
        assert akerr.geterrcall() is before

    def test_errstate_is_nested_and_restores_correctly(self):
        base = akerr.geterr()
        with akerr.errstate(divide="warn"):
            assert akerr.geterr()["divide"] == "warn"
            with akerr.errstate(divide="raise"):
                assert akerr.geterr()["divide"] == "raise"
            # inner restored
            assert akerr.geterr()["divide"] == "warn"
        # outer restored
        assert akerr.geterr() == base

    # ---------- handle() dispatch behavior ----------

    def test_handle_ignore_mode_emits_nothing(self, capsys):
        akerr.seterr(divide="ignore")
        # Should neither warn, raise, nor print/log
        akerr.handle("divide", "msg")
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_handle_warn_mode_emits_runtimewarning(self):
        akerr.seterr(divide="warn")
        with pytest.warns(RuntimeWarning):
            akerr.handle("divide", "something bad")

    def test_handle_raise_mode_raises_fpe(self):
        akerr.seterr(divide="raise")
        with pytest.raises(FloatingPointError):
            akerr.handle("divide", "boom")

    def test_handle_call_mode_invokes_handler(self):
        seen = []

        def handler(kind, msg):
            seen.append((kind, msg))

        akerr.seterr(divide="call")
        akerr.seterrcall(handler)
        akerr.handle("divide", "hello")
        assert seen and seen[0][0] == "divide" and "hello" in seen[0][1]

    def test_handle_call_mode_with_no_handler_is_noop(self):
        akerr.seterr(divide="call")
        akerr.seterrcall(None)
        # Should not raise
        akerr.handle("divide", "no handler")

    def test_handle_print_mode_writes_stdout(self, capsys):
        akerr.seterr(divide="print")
        akerr.handle("divide", "printed")
        out = capsys.readouterr().out
        # Module writes without newline
        assert out == "divide: printed"

    def test_handle_log_mode_logs_error(self):
        from arkouda.numpy import err as akerr

        # Use the exact logger the module uses
        from arkouda.numpy.err import (
            _logger as module_logger,  # test-only import of private symbol
        )

        class ListHandler(logging.Handler):
            def __init__(self, level=logging.ERROR):
                super().__init__(level)
                self.records = []

            def emit(self, record):
                self.records.append(record)

        # Save current logger state and attach our own handler
        old_handlers = module_logger.handlers[:]
        old_level = module_logger.level
        old_propagate = module_logger.propagate
        h = ListHandler(logging.ERROR)

        try:
            module_logger.handlers = [h]  # ensure our handler sees it
            module_logger.setLevel(logging.ERROR)
            module_logger.propagate = False  # avoid double-emitting to parents

            akerr.seterr(divide="log")
            akerr.handle("divide", "logged")

            assert any("divide: logged" in r.getMessage() for r in h.records), (
                f"got records: {[r.getMessage() for r in h.records]}"
            )
        finally:
            # restore logger
            module_logger.handlers = old_handlers
            module_logger.setLevel(old_level)
            module_logger.propagate = old_propagate

    def test_handle_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            akerr.handle("not-a-kind", "msg")  # type: ignore[arg-type]

    def _make_operands(
        self, op: str, mode: Literal["pdarrays", "left_scalar", "right_scalar"] = "pdarrays"
    ):
        """
        Build (numerator, denominator) ensuring a zero is present in the denominator side
        that the pdarray method will inspect before performing the op.

        We use float arrays so `/` and `//` yield inf instead of a backend hard error.
        """
        # Will inspect the right-hand side for __truediv__/__floordiv__,
        # and self (the pdarray instance) for __rtruediv__/__rfloordiv__.
        a = ak.array([1.0, 2.0])  # nonzero
        b = ak.array([0.0, 2.0])  # has a zero

        if mode == "pdarrays":
            numerator, denominator = a, b
        elif mode == "right_scalar":
            numerator, denominator = a, 0.0
        elif mode == "left_scalar":
            numerator, denominator = 1.0, b
        else:
            raise ValueError("mode must be pdarrays, right_scalar, or left_scalar.")

        if op == "/":

            def do(n, d):
                return n / d
        elif op == "//":

            def do(n, d):
                return n // d
        else:
            raise ValueError(op)

        return numerator, denominator, do

    # --- tests: ignore (no warning/exception) ----------------------------------

    @pytest.mark.parametrize("op", ["/", "//"])
    @pytest.mark.parametrize("mode", ["pdarrays", "left_scalar", "right_scalar"])
    def test_divide_ignore_allows_operation(self, op, mode):
        num, den, do = self._make_operands(op, mode=mode)
        with akerr.errstate(divide="ignore"):
            out = do(num, den)  # should not warn or raise
        # sanity: we get inf where division by zero occurs
        is_inf = ak.isinf(out) if op == "/" else ak.isinf(out)  # both ops should produce inf for 1/0.0
        assert bool(ak.any(is_inf))

    # --- tests: warn (RuntimeWarning) ------------------------------------------

    @pytest.mark.parametrize("op", ["/", "//"])
    @pytest.mark.parametrize("mode", ["pdarrays", "left_scalar", "right_scalar"])
    def test_divide_warn_emits_runtimewarning(self, op, mode):
        num, den, do = self._make_operands(op, mode=mode)
        with akerr.errstate(divide="warn"):
            with pytest.warns(RuntimeWarning):
                _ = do(num, den)

    # --- tests: raise (FloatingPointError) -------------------------------------

    @pytest.mark.parametrize("op", ["/", "//"])
    @pytest.mark.parametrize("mode", ["pdarrays", "left_scalar", "right_scalar"])
    def test_divide_raise_throws_floatingpointerror(self, op, mode):
        num, den, do = self._make_operands(op, mode=mode)
        with akerr.errstate(divide="raise"):
            with pytest.raises(FloatingPointError):
                _ = do(num, den)

    # --- tests: call (custom handler invoked) ----------------------------------

    @pytest.mark.parametrize("op", ["/", "//"])
    @pytest.mark.parametrize("mode", ["pdarrays", "left_scalar", "right_scalar"])
    def test_divide_call_invokes_handler(self, op, mode):
        num, den, do = self._make_operands(op, mode=mode)
        seen = []

        def handler(kind: str, msg: str):
            # Arkouda's stub uses (kind, message) signature
            seen.append((kind, msg))

        prev = akerr.seterrcall(handler)
        try:
            with akerr.errstate(divide="call"):
                _ = do(num, den)
            # At least one event should have been sent to the handler
            assert seen, "custom error handler was not invoked"
            # Kind should be "divide"
            assert all(k == "divide" for (k, _m) in seen)
            # Message should include a standard phrase
            assert any("divide by zero encountered" in m for (_k, m) in seen)
        finally:
            akerr.seterrcall(prev)
