from __future__ import annotations

import io
import sys
import warnings

import pytest

import arkouda as ak


# --- helpers ---------------------------------------------------------------

ERR_KINDS = ("divide", "over", "under", "invalid")
MODES = ("ignore", "warn", "raise", "call", "print", "log")


def _reset_ak_err_to_defaults() -> None:
    # Arkouda scaffold defaults mirror NumPy defaults per module docstring.
    ak.seterr(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    ak.seterrcall(None)


@pytest.fixture(autouse=True)
def _clean_errstate():
    _reset_ak_err_to_defaults()
    yield
    _reset_ak_err_to_defaults()


# --- tests ----------------------------------------------------------------


@pytest.mark.parametrize("kind", ERR_KINDS)
@pytest.mark.parametrize("mode", MODES)
def test_seterr_roundtrip_and_return_previous(kind: str, mode: str) -> None:
    """Seterr returns the previous dict and updates the requested key."""
    before = ak.geterr()
    prev = ak.seterr(**{kind: mode})
    after = ak.geterr()

    assert prev == before
    assert after[kind] == mode
    # other keys unchanged
    for k in ERR_KINDS:
        if k != kind:
            assert after[k] == before[k]


def test_errstate_restores_state() -> None:
    ak.seterr(divide="raise")
    assert ak.geterr()["divide"] == "raise"

    with ak.errstate(divide="warn"):
        assert ak.geterr()["divide"] == "warn"

    # restored
    assert ak.geterr()["divide"] == "raise"


def test_errstate_nested_restores_correctly() -> None:
    ak.seterr(divide="ignore")
    with ak.errstate(divide="warn"):
        assert ak.geterr()["divide"] == "warn"
        with ak.errstate(divide="raise"):
            assert ak.geterr()["divide"] == "raise"
        assert ak.geterr()["divide"] == "warn"
    assert ak.geterr()["divide"] == "ignore"


def test_seterrcall_geterrcall_roundtrip() -> None:
    def handler(kind: str, msg: str) -> None:
        pass

    assert ak.geterrcall() is None
    prev = ak.seterrcall(handler)
    assert prev is None
    assert ak.geterrcall() is handler

    prev2 = ak.seterrcall(None)
    assert prev2 is handler
    assert ak.geterrcall() is None


def test_errstate_temporarily_sets_call_handler() -> None:
    calls: list[tuple[str, str]] = []

    def handler(kind: str, msg: str) -> None:
        calls.append((kind, msg))

    ak.seterr(divide="call")
    assert ak.geterrcall() is None

    with ak.errstate(call=handler):
        assert ak.geterrcall() is handler
        ak.numpy.err.handle("divide", "divide by zero encountered")

    assert calls == [("divide", "divide by zero encountered")]
    # restored
    assert ak.geterrcall() is None


@pytest.mark.parametrize("kind", ERR_KINDS)
def test_handle_known_kinds_do_not_raise(kind: str) -> None:
    # sanity: known kinds do not raise at validation layer
    ak.numpy.err.handle(kind, "msg")


def test_handle_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ak.numpy.err.handle("bogus", "nope")


def test_mode_ignore_no_side_effect() -> None:
    ak.seterr(divide="ignore")
    # Should do nothing; just ensure it doesn't raise or warn.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ak.numpy.err.handle("divide", "divide by zero encountered")
        assert len(w) == 0


def test_mode_warn_emits_runtimewarning() -> None:
    ak.seterr(divide="warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ak.numpy.err.handle("divide", "divide by zero encountered")
        assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_mode_raise_raises_floating_point_error() -> None:
    ak.seterr(divide="raise")
    with pytest.raises(FloatingPointError):
        ak.numpy.err.handle("divide", "divide by zero encountered")


def test_mode_call_invokes_handler() -> None:
    seen: list[tuple[str, str]] = []

    def handler(kind: str, msg: str) -> None:
        seen.append((kind, msg))

    ak.seterr(divide="call")
    ak.seterrcall(handler)

    ak.numpy.err.handle("divide", "divide by zero encountered")
    assert seen == [("divide", "divide by zero encountered")]


def test_mode_print_writes_to_stdout() -> None:
    ak.seterr(divide="print")
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        ak.numpy.err.handle("divide", "divide by zero encountered")
    finally:
        sys.stdout = old
    assert "divide: divide by zero encountered" in buf.getvalue()


def test_mode_log_does_not_crash() -> None:
    # We don't assert logger output here; just that it routes without error.
    ak.seterr(divide="log")
    ak.numpy.err.handle("divide", "divide by zero encountered")


@pytest.mark.parametrize("kind", ("over", "under", "invalid"))
def test_seterr_warns_for_unimplemented_kinds(kind: str) -> None:
    """
    err.py warns that over/under/invalid are not implemented yet.
    We assert a warning is emitted when changing those from current value.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ak.seterr(**{kind: "warn"})
        assert any("not implemented yet" in str(x.message) for x in w)
