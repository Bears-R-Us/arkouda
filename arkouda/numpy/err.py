"""
Floating-point error state management for Arkouda.

This module provides a NumPy-like API for controlling how Arkouda handles
floating-point errors such as divide-by-zero, overflow, underflow, and
invalid operations. The design mirrors `numpy.errstate`, `numpy.seterr`,
and related functions, so users familiar with NumPy can expect similar
semantics.

Unlike NumPy, Arkouda computations occur in a distributed backend, so
this module currently acts as a lightweight *scaffold*: it records user
preferences and dispatches to Python-side handlers when explicitly
invoked by front-end code. By default, there is **no performance cost**,
since core numerical kernels do not consult this state unless explicitly
wired to do so.

Available error modes
---------------------
Each error category ('divide', 'over', 'under', 'invalid') may be set to
one of the following modes:

- ``"ignore"`` : silently ignore the error (default).
- ``"warn"``   : issue a Python ``RuntimeWarning``.
- ``"raise"``  : raise a ``FloatingPointError`` exception.
- ``"call"``   : invoke a user-supplied callable (see :func:`seterrcall`).
- ``"print"``  : write a message to standard output.
- ``"log"``    : log a message via Arkouda's logger.

API
---
- :func:`geterr`       : return the current error handling settings.
- :func:`seterr`       : set global error handling, returning the old settings.
- :func:`geterrcall`   : get the callable used for ``"call"`` mode.
- :func:`seterrcall`   : set the callable used for ``"call"`` mode.
- :func:`errstate`     : context manager to temporarily change settings.
- :func:`handle`       : route an error condition through the current policy.

Example
-------
>>> import arkouda as ak
>>> ak.geterr()
{'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}

>>> with ak.errstate(divide="warn"):
...     out = ak.array([1.0]) / 0

>>> def myhandler(kind, msg): print(f"[ak] {kind}: {msg}")
>>> ak.seterr(divide="call")
{'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
>>> ak.seterrcall(myhandler)
>>> out = ak.array([1.0]) / 0
[ak] divide: divide by zero encountered
"""

from __future__ import annotations

import contextlib
import sys

# --- add near the top of arkouda/err.py ---
import warnings

from typing import Callable, Dict, Iterator, Literal, Optional

from arkouda.core.logger import get_arkouda_logger


__all__ = [
    "geterr",
    "seterr",
    "geterrcall",
    "seterrcall",
    "errstate",
]

_ErrorMode = Literal["ignore", "warn", "raise", "call", "print", "log"]

# Default modes chosen to mirror NumPy's defaults
_default_errstate: Dict[str, _ErrorMode] = {
    "divide": "ignore",
    "over": "ignore",
    "under": "ignore",
    "invalid": "ignore",
}

# Global state (lightweight; consulted only if you wire checks later)
_ak_errstate: Dict[str, _ErrorMode] = dict(_default_errstate)
_ak_errcall: Optional[Callable[[str, str], None]] = None  # (errtype, message) -> None


_logger = get_arkouda_logger("arkouda.err")


def _dispatch(kind: str, message: str) -> None:
    """
    Perform the side-effect for an FP error according to the current mode.
    No-ops unless explicitly invoked (e.g., via postcheck_* helpers).
    """
    mode = _ak_errstate.get(kind, "ignore")

    if mode == "ignore":
        return
    if mode == "warn":
        warnings.warn(f"{kind}: {message}", RuntimeWarning, stacklevel=3)
        return
    if mode == "raise":
        raise FloatingPointError(f"{kind}: {message}")
    if mode == "call":
        if _ak_errcall is not None:
            _ak_errcall(kind, message)
        return
    if mode == "print":
        sys.stdout.write(f"{kind}: {message}")
        return
    if mode == "log":
        _logger.error("%s: %s", kind, message)
        return


def handle(kind: str, message: str) -> None:
    """Public hook so other modules (or users) can route an error through the policy."""
    if kind not in _default_errstate:
        raise ValueError(f"Unknown error type: {kind!r}")
    _dispatch(kind, message)


def geterr() -> Dict[str, _ErrorMode]:
    """
    Get the current Arkouda floating-point error handling settings.

    Returns
    -------
    dict
        Mapping of {'divide', 'over', 'under', 'invalid'} to one of
        {'ignore', 'warn', 'raise', 'call', 'print', 'log'}.
    """
    return _ak_errstate.copy()


def seterr(**kwargs: _ErrorMode) -> Dict[str, _ErrorMode]:
    """
    Set how Arkouda handles floating-point errors.

    Parameters
    ----------
    divide, over, under, invalid : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Behavior for the corresponding error category.

    Returns
    -------
    dict
        The previous settings.

    Notes
    -----
    This is a *scaffold* API. It does not change backend behavior yet; it only
    records the desired policy so future operations can consult it.
    """
    old = _ak_errstate.copy()
    valid: set[_ErrorMode] = {"ignore", "warn", "raise", "call", "print", "log"}

    for key, val in kwargs.items():
        if key in ["over", "under", "invalid"] and val != _ak_errstate[key]:
            warnings.warn(f"{key} error type is not implemented yet and therefore will have no effect.")
        if key not in _default_errstate:
            raise ValueError(f"Unknown error type: {key!r}")
        if val not in valid:
            raise ValueError(f"Invalid setting {val!r} for {key!r}")
        _ak_errstate[key] = val
    return old


def geterrcall() -> Optional[Callable[[str, str], None]]:
    """
    Get the current callable used when error mode is 'call'.

    Returns
    -------
    callable or None
        A function of signature (errtype: str, message: str) -> None, or None.
    """
    return _ak_errcall


def seterrcall(func: Optional[Callable[[str, str], None]]) -> Optional[Callable[[str, str], None]]:
    """
    Set the callable invoked when an error category is set to 'call'.

    Parameters
    ----------
    func : callable or None
        Function of signature (errtype: str, message: str) -> None.
        Pass None to clear.

    Returns
    -------
    callable or None
        The previous callable.

    Notes
    -----
    This is a *stub* for API compatibility. Arkouda does not currently invoke
    this callable; it is stored for future use.
    """
    global _ak_errcall
    old = _ak_errcall
    if func is not None and not callable(func):
        raise TypeError("seterrcall expects a callable or None")
    _ak_errcall = func
    return old


@contextlib.contextmanager
def errstate(
    *,
    divide: Optional[_ErrorMode] = None,
    over: Optional[_ErrorMode] = None,
    under: Optional[_ErrorMode] = None,
    invalid: Optional[_ErrorMode] = None,
    call: Optional[Callable[[str, str], None]] = None,
) -> Iterator[None]:
    """
    Context manager to temporarily set floating-point error handling.

    Parameters
    ----------
    divide, over, under, invalid : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Temporary behavior within the context.
    call : callable or None, optional
        Temporary callable used if any category is set to 'call'.
        Signature: (errtype: str, message: str) -> None.

    Yields
    ------
    None
        This context manager does not return a value. Code inside the ``with``
        block will be executed with the temporary error handling settings.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.geterr()
    {'divide': 'call', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> with ak.errstate(divide='warn'):
    ...     _ = ak.array([1.0]) / 0  # gives a warning
    >>> ak.geterr()['divide']   # doctest: +SKIP
    'ignore'

    Notes
    -----
    This affects only stored policy; it does not add runtime checks by itself.
    """
    # Save old state
    old_modes = geterr()
    old_call = geterrcall()

    # Apply temporary modes
    updates = {}
    if divide is not None:
        updates["divide"] = divide
    if over is not None:
        updates["over"] = over
    if under is not None:
        updates["under"] = under
    if invalid is not None:
        updates["invalid"] = invalid
    if updates:
        seterr(**updates)
    if call is not None:
        seterrcall(call)

    try:
        yield
    finally:
        # Restore previous state
        seterr(**old_modes)
        seterrcall(old_call)
