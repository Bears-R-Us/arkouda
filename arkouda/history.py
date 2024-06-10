import readline
from IPython.core.history import HistoryAccessor
from typing import List, Optional


class HistoryRetriever:
    """
    HistoryRetriever is an abstract base class that defines the retrieve method signature
    and implements _filter_arkouda_command
    """

    def _filter_arkouda_command(self, command: str, filter_string: str = "ak") -> Optional[str]:
        """
        Returns command string if the filter string is in the command and the
        command is not generate_history. Otherwise, returns None
        """
        return command if (filter_string in command and "generate_history" not in command) else None

    def retrieve(
        self, command_filter: Optional[str] = None, num_commands: Optional[int] = None
    ) -> List[str]:
        """
        Generates list of commands executed within a Python REPL shell, Jupyter notebook,
        or IPython notebook, with an optional command filter and number of commands to return.

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
        """
        raise NotImplementedError("Derived classes must implement retrieve")


class ShellHistoryRetriever(HistoryRetriever):
    """
    ShellHistoryRetriever implements the retrieve method to get command history from the
    Python REPL shell.
    """

    def retrieve(
        self, command_filter: Optional[str] = None, num_commands: Optional[int] = None
    ) -> List[str]:
        """
        Generates list of commands executed within the a Python REPL shell, with an
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
    """
    NotebookHistoryRetriever implements the retrieve method to get command history
    from a Jupyter notebook or IPython shell.
    """

    def retrieve(
        self, command_filter: Optional[str] = None, num_commands: Optional[int] = None
    ) -> List[str]:
        """
        Generates list of commands executed within a Jupyter notebook or IPython shell,
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
        """
        raw = True  # HistoryAccessor _run_sql method parameter
        output = False  # HistoryAccessor _run_sql method parameter
        n = 0  # HistoryAccessor _run_sql method parameter
        num_to_return = num_commands if num_commands else 100

        if n is None:
            cur = self._run_sql("ORDER BY session DESC, line DESC LIMIT ?", (n,), raw=raw, output=output)
        else:
            cur = self._run_sql("ORDER BY session DESC, line DESC", (), raw=raw, output=output)

        if command_filter:
            return [
                cmd[2]
                for cmd in reversed(list(cur))
                if self._filter_arkouda_command(cmd[2], command_filter)
            ][-num_to_return:]
        else:
            return [cmd[2] for cmd in reversed(list(cur))][-num_to_return:]
