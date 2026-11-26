"""File publication survives Windows readers without losing the previous file."""
import ctypes
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tools import atomic_files as files


@pytest.mark.parametrize("kind", ["json", "npz"])
def test_transient_sharing_lock_retries_without_partial_publication(tmp_path, monkeypatch, kind):
    path = tmp_path / ("data." + kind)
    path.write_bytes(b"previous complete file")
    replace = files.os.replace
    attempts = []

    def locked(source, target):
        attempts.append(source)
        if len(attempts) < 3:
            assert path.read_bytes() == b"previous complete file"
            raise PermissionError(13, "file is being used by another process")
        replace(source, target)

    monkeypatch.setattr(files.os, "replace", locked)
    monkeypatch.setattr(files.time, "sleep", lambda delay: None)
    if kind == "json":
        files.atomic_json(path, {"version": 2})
        assert json.loads(path.read_text()) == {"version": 2}
    else:
        files.atomic_write(path, lambda stream: np.savez_compressed(stream, X=np.arange(4)))
        with np.load(path) as data:
            np.testing.assert_array_equal(data["X"], np.arange(4))
    assert len(attempts) == 3
    assert list(tmp_path.iterdir()) == [path]


def test_persistent_lock_preserves_old_file_and_reports_failure(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    files.atomic_json(path, {"version": 1})

    def locked(source, target):
        raise PermissionError(13, "locked")

    monkeypatch.setattr(files.os, "replace", locked)
    monkeypatch.setattr(files.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        files.atomic_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_overlapping_writers_have_private_files(tmp_path):
    path = tmp_path / "data.json"
    barrier = threading.Barrier(2)

    def publish(value):
        def write(stream):
            stream.write(json.dumps({"value": value}).encode())
            barrier.wait(timeout=5)
        files.atomic_write(path, write)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(publish, v) for v in (1, 2)]
        for job in jobs:
            job.result(timeout=10)
    assert json.loads(path.read_text()) in ({"value": 1}, {"value": 2})
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_real_windows_reader_lock_can_release_during_publication(tmp_path):
    from ctypes import wintypes

    path = tmp_path / "data.json"
    files.atomic_json(path, {"version": 1})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Allow reading/writing but withhold FILE_SHARE_DELETE, like ordinary readers.
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value

    def release():
        time.sleep(.12)
        kernel.CloseHandle(handle)

    thread = threading.Thread(target=release)
    thread.start()
    try:
        files.atomic_json(path, {"version": 2})
    finally:
        thread.join(timeout=3)
    assert json.loads(path.read_text()) == {"version": 2}
