#!/usr/bin/env bash
# stop.sh <data dir>: stop the camelid serve started by serve.sh for that data dir, by its recorded pid only.
set -u
PIDFILE="$1/server.pid"
[ -f "$PIDFILE" ] || { echo "no pid file"; exit 0; }
PID=$(cat "$PIDFILE")
# argv[0] is any camelid binary (camelid, camelid-new, camelid-base), argv[1] is "serve".
ARGV0=$(tr '\0' '\n' < "/proc/$PID/cmdline" 2>/dev/null | sed -n 1p)
ARGV1=$(tr '\0' '\n' < "/proc/$PID/cmdline" 2>/dev/null | sed -n 2p)
case "$(basename "$ARGV0")" in
  camelid*) IS_CAMELID=1 ;;
  *) IS_CAMELID=0 ;;
esac
if [ "$IS_CAMELID" = 1 ] && [ "$ARGV1" = serve ]; then
  kill "$PID"
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$PID" 2>/dev/null; then echo "pid $PID did not stop" >&2; exit 1; fi
  echo "stopped $PID"
else
  echo "pid $PID is not a running camelid serve"
fi
rm -f "$PIDFILE"
