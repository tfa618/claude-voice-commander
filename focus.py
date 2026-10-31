"""
Detect which application is in the foreground, so the app only listens while
Claude Desktop is the active window. Uses only built-in OS tools:

  Windows : Win32 API via ctypes (no extra install)
  macOS   : `lsappinfo` (ships with macOS, needs no extra permission)
  Linux   : `xdotool` (X11 only; install with your package manager)

If detection isn't possible, get_foreground_app() returns None and the caller
decides what to do (main.py treats "unknown" as "allowed" and warns once).
"""
import os
import platform
import re
import subprocess
from typing import Optional

_SYSTEM = platform.system()


def app_matches(app_name: Optional[str], targets: list[str]) -> bool:
    """
    True if the foreground app name matches one of the target names.
    Matching is case-insensitive on whole words of the app name, so "claude"
    matches "Claude" and "claude.exe" but not "claudette".
    """
    if not app_name:
        return False
    name = app_name.lower()
    name = re.sub(r"\.(exe|app)$", "", name.strip())
    words = set(re.split(r"[^a-z0-9]+", name))
    for t in targets:
        t = t.lower().strip()
        if not t:
            continue
        if name == t or t in words:
            return True
    return False


def _windows_foreground() -> Optional[str]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    # ctypes assumes every return value is a 32-bit int. HWND/HANDLE are
    # pointer-sized on 64-bit Windows, so declare the real types or handles
    # can be truncated and detection fails (which would mean "always listen").
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
    finally:
        kernel32.CloseHandle(handle)
    return None


def _mac_foreground() -> Optional[str]:
    asn = subprocess.run(["lsappinfo", "front"], capture_output=True, text=True, timeout=2).stdout.strip()
    if not asn:
        return None
    out = subprocess.run(["lsappinfo", "info", "-only", "name", asn],
                         capture_output=True, text=True, timeout=2).stdout
    m = re.search(r'"(?:LSDisplayName|name)"\s*=\s*"([^"]*)"', out)
    return m.group(1) if m else None


def _linux_foreground() -> Optional[str]:
    pid = subprocess.run(["xdotool", "getactivewindow", "getwindowpid"],
                         capture_output=True, text=True, timeout=2).stdout.strip()
    if pid.isdigit():
        try:
            with open(f"/proc/{pid}/comm", "r") as f:
                return f.read().strip()
        except OSError:
            pass
    title = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                           capture_output=True, text=True, timeout=2).stdout.strip()
    return title or None


def get_foreground_app() -> Optional[str]:
    """Name of the foreground app/process, or None if it can't be determined."""
    try:
        if _SYSTEM == "Windows":
            return _windows_foreground()
        if _SYSTEM == "Darwin":
            return _mac_foreground()
        return _linux_foreground()
    except Exception:
        return None
