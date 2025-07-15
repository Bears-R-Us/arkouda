"""
History retrieval utilities for Arkouda command execution.

This module provides tools for retrieving the history of commands executed
in Python REPL shells or Jupyter/IPython notebooks. It defines abstract and
concrete retrievers to access interactive command history for reproducibility,
debugging, or audit purposes.

Classes
-------
HistoryRetriever
    Abstract base class defining the `retrieve` method and a helper for filtering commands.

ShellHistoryRetriever
    Retrieves command history from a Python REPL shell using `readline`.

NotebookHistoryRetriever
    Retrieves command history from a Jupyter notebook or IPython shell via IPython's history database.

Usage
-----
Used internally by `arkouda.generate_history()` to provide a user-friendly interface
for querying and filtering past commands based on optional string filters and count limits.

Examples
--------
>>> from arkouda.history import ShellHistoryRetriever, NotebookHistoryRetriever

# REPL mode
>>> h = ShellHistoryRetriever()
>>> h.retrieve(command_filter="ak.", num_commands=5)   # doctest: +SKIP
['ak.array([1,2,3])', 'ak.sum(...)', ...]

# Notebook mode
>>> h = NotebookHistoryRetriever()  # doctest: +SKIP
>>> h.retrieve(num_commands=3)  # doctest: +SKIP
['ak.connect()', 'df = ak.DataFrame(...)', 'ak.argsort(...)']

See Also
--------
arkouda.generate_history : High-level function for retrieving command history.

"""

import readline
from typing import List, Optional

from IPython.core.history import HistoryAccessor

__all__ = [
    "HistoryRetriever",
    "NotebookHistoryRetriever",
    "ShellHistoryRetriever",
]


class HistoryRetriever:
    """
    Abstract base class that defines the retrieve method signature.

    Implements _filter_arkouda_command.
    """

    def _filter_arkouda_command(self, command: str, filter_string: str = "ak") -> Optional[str]:
        """
        Return command string.

        Return command string if the filter string is in the command and the
        command is not generate_history. Otherwise, returns None
        """
        return command if (filter_string in command and "generate_history" not in command) else None

    def retrieve(
        self, command_filter: Optional[str] = None, num_commands: Optional[int] = None
    ) -> List[str]:
        """
        Generate list of commands executed.

        Generate list of commands executed within a Python REPL shell, Jupyter notebook,
        or IPython notebook, with an optional command filter and number of commands to return.

        Parameters
        ----------
        num_commands : int
            The number of commands from history to retrieve
        command_filter : str
            String containing characters used to select a subset of command history.

        Returns  # noqa: DAR202
        -------
        List[str]
            A list of commands from the Python shell, Jupyter notebook, or IPython notebook

        """
        raise NotImplementedError("Derived classes must implement retrieve")


class ShellHistoryRetriever(HistoryRetriever):
    """Implement the retrieve method to get command history from the Python REPL shell."""

    def retrieve(
        self, command_filter: Optional[str] = None, num_commands: Optional[int] = None
    ) -> List[str]:
        """
        Generate list of commands executed.

        Generate list of commands executed within the a Python REPL shell, with an
        optional command filter and number of commands to return.

        Parameters
        ----------
        num_commands : int
            The number of commands from history to retrieve
        command_filter : str
            String containing characters used to select a subset of command history.

        Returns
        -------
        List[str]
            A list of commands from the Python shell, Jupyter notebook, or IPython notebook

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.history import ShellHistoryRetriever
        >>> import readline
        >>> h = ShellHistoryRetriever()
        >>> readline.clear_history()
        >>> 1 + 2
        3
        >>> h.retrieve() # doctest: +SKIP
        [' 1 + 2', 'h.retrieve()']

        """
        length_of_history = readline.get_current_history_length()
        num_to_return = num_commands if num_commands else length_of_history

        if command_filter:
            return [
                readline.get_history_item(i + 1)
                for i in range(length_of_history)
                if self._filter_arkouda_command(readline.get_history_item(i + 1), command_filter)
            ][-num_to_return:]
        else:
            return [str(readline.get_history_item(i + 1)) for i in range(length_of_history)][
                -num_to_return:
            ]


class NotebookHistoryRetriever(HistoryAccessor, HistoryRetriever):
    """Implement the retrieve method to get command history from a Jupyter notebook or IPython shell."""

    def retrieve(
        self, command_filter: Optional[str] = None, num_commands: Optional[int] = None
    ) -> List[str]:
        """
        Generate list of commands executed.

        Generate list of commands executed within a Jupyter notebook or IPython shell,
        with an optional command filter and number of commands to return.

        Parameters
        ----------
        num_commands : int
            The number of commands from history to retrieve
        command_filter : str
            String containing characters used to select a subset of command history.

        Returns
        -------
        List[str]
            A list of commands from the Python shell, Jupyter notebook, or IPython notebook

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.history import NotebookHistoryRetriever
        >>> h = NotebookHistoryRetriever()  # doctest: +SKIP
        >>> 1+2
        3
        >>> 4*6
        24
        >>> 2**3
        8
        >>> h.retrieve(num_commands=3)  # doctest: +SKIP
        ['1+2', '4*6', '2**3']

        """
        raw = True  # HistoryAccessor _run_sql method parameter
        output = False  # HistoryAccessor _run_sql method parameter
        n = 0  # HistoryAccessor _run_sql method parameter
        num_to_return = num_commands if num_commands else 100

        if n is None:
            cur = self._run_sql("ORDER BY session DESC, line DESC LIMIT ?", (n,), raw=raw, output=output)
        else:
            cur = self._run_sql("ORDER BY session DESC, line DESC", (), raw=raw, output=output)

        ret = [cmd[2] for cmd in reversed(list(cur)) if isinstance(cmd[2], str)]
        if command_filter:
            ret = [cmd for cmd in ret if self._filter_arkouda_command(cmd, command_filter)]

        return ret[-num_to_return:]
