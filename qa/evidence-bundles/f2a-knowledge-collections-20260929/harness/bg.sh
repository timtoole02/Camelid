#!/usr/bin/env bash
# Run a command fully detached from the ssh session: bg.sh <log file> <command...>
log="$1"
shift
setsid nohup bash -c "source ~/.cargo/env; $*; echo EXIT=\$? >> '$log'" > "$log" 2>&1 < /dev/null &
echo "started pid $!"
