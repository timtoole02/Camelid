#!/usr/bin/env python3
"""Stream one regular file through SHA-256 without warming macOS's file cache."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import sys
from pathlib import Path


F_NOCACHE = 48
READ_BYTES = 4 * 1024 * 1024


def sha256_nocache(path: Path) -> str:
    if sys.platform != "darwin":
        raise RuntimeError("F_NOCACHE hashing is supported only on macOS")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        identity_before = os.fstat(descriptor)
        if not stat.S_ISREG(identity_before.st_mode):
            raise RuntimeError(f"input is not a regular file: {path}")
        fcntl.fcntl(descriptor, F_NOCACHE, 1)
        while block := os.read(descriptor, READ_BYTES):
            digest.update(block)
            bytes_read += len(block)
        identity_after = os.fstat(descriptor)
        stable_identity = (
            identity_before.st_dev,
            identity_before.st_ino,
            identity_before.st_size,
            identity_before.st_mtime_ns,
        ) == (
            identity_after.st_dev,
            identity_after.st_ino,
            identity_after.st_size,
            identity_after.st_mtime_ns,
        )
        if not stable_identity or bytes_read != identity_before.st_size:
            raise RuntimeError(f"input identity changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: sha256_nocache.py ABSOLUTE_FILE", file=sys.stderr)
        return 64
    path = Path(argv[1])
    if not path.is_absolute():
        print("REFUSED: hash input must be absolute", file=sys.stderr)
        return 75
    try:
        print(sha256_nocache(path))
    except (OSError, RuntimeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
