#!/usr/bin/env python3
"""Capture one schema-bound host-memory sample with the canonical watchdog."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class CaptureError(RuntimeError):
    """The canonical telemetry source or output contract was unavailable."""


def _load_watchdog(path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"watchdog is not a regular non-symlink file: {path}")
    specification = importlib.util.spec_from_file_location("hot48_watchdog", path)
    if specification is None or specification.loader is None:
        raise CaptureError(f"could not load watchdog telemetry source: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _boot_identity() -> str:
    result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.boottime"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value:
        raise CaptureError("kern.boottime returned an empty identity")
    return value


def capture(watchdog_path: Path) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise CaptureError("host-memory capture is supported only on macOS")
    watchdog = _load_watchdog(watchdog_path)
    telemetry_type = getattr(watchdog, "NativeTelemetry", None)
    if telemetry_type is None:
        raise CaptureError("watchdog does not expose NativeTelemetry")
    host = telemetry_type().sample_host()
    if not isinstance(host, dict):
        raise CaptureError("watchdog host sample is not an object")
    return {
        "schema_version": 1,
        "telemetry_source": "run_load_only_watchdog.NativeTelemetry.sample_host",
        "boot_identity": _boot_identity(),
        "host": host,
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise CaptureError("output path must be absolute")
    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to replace existing output: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchdog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _write_exclusive(args.output, capture(args.watchdog))
    except (CaptureError, OSError, subprocess.SubprocessError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
