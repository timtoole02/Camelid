#!/usr/bin/env bash
# Does unloading a model actually free its weights on macOS?
#
# `ps -o rss` is the WRONG instrument here: the weights are page-aligned wire pages the
# GPU reads in place, and an untouched model never faults them in (a 3.4 GB model showed
# 189 MiB RSS). So each model is EXERCISED with a real generation to force its pages
# resident, and the measurement is `footprint`'s physical footprint, which counts them.
set -uo pipefail
BIN="$1"; LABEL="$2"; PORT="$3"
A="$HOME/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
B="$HOME/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
OUT="$HOME/camelid-memctx/evict-$LABEL"; mkdir -p "$OUT"

"$BIN" serve --addr 127.0.0.1:$PORT >"$OUT/serve.out" 2>"$OUT/serve.err" &
SRV=$!
for _ in $(seq 1 180); do curl -s "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break; sleep 1; done

fp() { footprint -p "$SRV" 2>/dev/null | grep -iE "phys_footprint|Physical footprint" | tail -1 | grep -oE "[0-9.]+ *[KMG]B?" | tail -1; }
say() { printf '%-30s %14s\n' "$1" "$(fp)"; }

load() {  # exercise the model so its wire pages are actually faulted in
  curl -s -X POST "http://127.0.0.1:$PORT/api/models/load" -H 'Content-Type: application/json' \
    -d "{\"path\":\"$1\",\"replace\":true}" >/dev/null 2>&1
  local m; m=$(curl -s "http://127.0.0.1:$PORT/v1/models" | sed 's/.*"id":"\([^"]*\)".*/\1/')
  curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to five.\"}],\"max_tokens\":32,\"temperature\":0}" >/dev/null 2>&1
  sleep 3
}

say "0 empty server"
load "$A"; say "1 3B-Q8_0 loaded + used"
curl -s -X POST "http://127.0.0.1:$PORT/api/models/unload" -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1
sleep 5; say "2 after unload"
load "$B"; say "3 1B-Q8_0 loaded + used"
load "$A"; say "4 back to 3B-Q8_0 + used"
curl -s -X POST "http://127.0.0.1:$PORT/api/models/unload" -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1
sleep 5; say "5 after final unload"

kill $SRV 2>/dev/null; for _ in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 1; done; kill -9 $SRV 2>/dev/null
