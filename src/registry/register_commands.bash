if [ -z "$CHPL_HOME" ]; then
    # We need CHPL_HOME to find run-in-venv.bash. Try falling back to a
    # compiler in PATH.
    if output=$(chpl --print-bootstrap-commands); then
        eval "$output"
    else
        echo "Error: CHPL_HOME is not set" 1>&2
        exit 1
    fi
fi

exec $CHPL_HOME/util/config/run-in-venv-with-python-bindings.bash \
    python3 $1/register_commands.py $2 $3 $4
