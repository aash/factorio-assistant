from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from queue import Queue
from typing import Generator

from ahk import AHK

PIPE_ACCESS_DUPLEX = 0x00000003
PIPE_TYPE_BYTE = 0x00000000
PIPE_READMODE_BYTE = 0x00000000
PIPE_WAIT = 0x00000000
ERROR_PIPE_CONNECTED = 535

kernel32 = ctypes.windll.kernel32

kernel32.CreateNamedPipeW.argtypes = [
    ctypes.c_wchar_p,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.LPVOID,
]
kernel32.CreateNamedPipeW.restype = ctypes.wintypes.HANDLE

kernel32.ConnectNamedPipe.argtypes = [
    ctypes.wintypes.HANDLE,
    ctypes.wintypes.LPVOID,
]
kernel32.ConnectNamedPipe.restype = ctypes.wintypes.BOOL

kernel32.ReadFile.argtypes = [
    ctypes.wintypes.HANDLE,
    ctypes.c_char_p,
    ctypes.wintypes.DWORD,
    ctypes.POINTER(ctypes.wintypes.DWORD),
    ctypes.wintypes.LPVOID,
]
kernel32.ReadFile.restype = ctypes.wintypes.BOOL

kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
kernel32.CloseHandle.restype = ctypes.wintypes.BOOL

kernel32.GetLastError.argtypes = []
kernel32.GetLastError.restype = ctypes.wintypes.DWORD


def _parse_event(line: str) -> dict | None:
    if ":" not in line:
        return None
    prefix, value = line.split(":", 1)
    if prefix == "c":
        return {"type": "char", "value": value}
    elif prefix == "d":
        return {"type": "down", "value": value}
    elif prefix == "u":
        return {"type": "up", "value": value}
    return None


def _pipe_reader(handle: int, queue: Queue[dict]) -> None:
    buf = ctypes.create_string_buffer(4096)
    data = ""
    while True:
        bytes_read = ctypes.wintypes.DWORD()
        success = kernel32.ReadFile(handle, buf, 4096, ctypes.byref(bytes_read), None)
        if not success or bytes_read.value == 0:
            break
        data += buf.raw[: bytes_read.value].decode("utf-8", errors="replace")
        while "\n" in data:
            line, data = data.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            event = _parse_event(line)
            if event is not None:
                queue.put(event)


@contextmanager
def input_hook() -> Generator[Queue[dict], None, None]:
    """
    Context manager that spawns an AHK v2 InputHook process communicating
    keypresses over a Windows named pipe.

    Yields a Queue[dict] where each item is an event dict:
        {"type": "char", "value": "A"}  — typed character (Shift-aware)
        {"type": "down", "value": "Enter"}  — key down for non-char keys
        {"type": "up", "value": "Shift"}  — key up

    OnChar handles printable characters with correct Shift state.
    OnKeyDown handles special keys (Enter, Escape, Backspace, etc.).
    OnKeyUp handles all key releases.
    """
    ahk = AHK(version="v2")
    exe_path = ahk._transport._executable_path  # ty:ignore[unresolved-attribute]
    pipe_name = f"\\\\.\\pipe\\ahk_input_hook_{os.getpid()}"

    ahk_script = (
        r'''
#Requires AutoHotkey v2.0
Global pipe := ""
Loop {
    Try {
        pipe := FileOpen("'''
        + pipe_name
        + r'''", "w", "UTF-8")
        Break
    }
    Sleep(20)
}

ih := InputHook("V L0")
ih.OnChar := OnChar
ih.OnKeyDown := OnKeyDown
ih.OnKeyUp := OnKeyUp
ih.KeyOpt("{All}", "N")
ih.Start()

OnChar(hook, char) {
    SendMsg("c:" char)
}

OnKeyDown(hook, vk, sc) {
    key := GetKeyName(Format("vk{:x}sc{:x}", vk, sc))
    SendMsg("d:" key)
}

OnKeyUp(hook, vk, sc) {
    key := GetKeyName(Format("vk{:x}sc{:x}", vk, sc))
    SendMsg("u:" key)
}

SendMsg(msg) {
    global pipe
    pipe.WriteLine(msg)
    DllCall("FlushFileBuffers", "Ptr", pipe.Handle)
}
'''
    )

    pipe_handle = kernel32.CreateNamedPipeW(
        pipe_name,
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1,
        65536,
        65536,
        5000,
        None,
    )
    if pipe_handle == ctypes.wintypes.HANDLE(-1).value:
        raise ctypes.WinError()

    connected = threading.Event()

    def _connect():
        result = kernel32.ConnectNamedPipe(pipe_handle, None)
        if result == 0:
            error = kernel32.GetLastError()
            if error != ERROR_PIPE_CONNECTED:
                raise ctypes.WinError(error)
        connected.set()

    connect_thread = threading.Thread(target=_connect, daemon=True)
    connect_thread.start()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ahk", delete=False, encoding="utf-8"
    ) as f:
        f.write(ahk_script)
        script_path = f.name

    proc = subprocess.Popen(
        [exe_path, "/CP65001", "/ErrorStdOut", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if not connected.wait(timeout=10):
        proc.terminate()
        raise TimeoutError("Timed out waiting for AHK to connect to named pipe")

    key_queue: Queue[dict] = Queue()
    reader_thread = threading.Thread(target=_pipe_reader, args=(pipe_handle, key_queue), daemon=True)
    reader_thread.start()

    try:
        yield key_queue
    finally:
        proc.terminate()
        proc.wait(timeout=3)
        try:
            os.unlink(script_path)
        except OSError:
            pass
        kernel32.CloseHandle(pipe_handle)
        reader_thread.join()