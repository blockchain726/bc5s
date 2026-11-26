"""Publish complete research files, tolerating brief Windows sharing locks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


def atomic_write(path, writer):
    """Write through a private binary stream, then replace without exposing partial data.

    Readers and virus scanners can briefly prevent replacement on Windows. Keep
    the previous file intact and retry for a bounded time; a persistent failure
    still reaches the caller. Each writer owns its temporary file.
    """
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                     prefix="." + path.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(min(.025 * 2**attempt, .2))
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(data))
