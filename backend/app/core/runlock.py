"""PID lock so only one process executes pipeline stages for a workspace at a time.

The lock file holds the pid of the process that is running a job. A lock whose pid is no longer
alive is stale and ignored, which is how a crashed or killed run is recognised immediately."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

LOCK_NAME = "run.lock"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == 259  # STILL_ACTIVE
            return True
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    def __init__(self, directory: Path):
        self.path = Path(directory) / LOCK_NAME
        self.held = False

    def owner(self) -> Optional[int]:
        """Pid of a live owner, or None when there is no lock or it is stale."""
        try:
            pid = int(self.path.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            return None
        return pid if pid_alive(pid) else None

    def acquire(self) -> bool:
        owner = self.owner()
        if owner is not None and owner != os.getpid():
            return False
        if owner == os.getpid():
            return True  # already held by this process (outer lock): the outer holder releases it
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        self.held = True
        return True

    def release(self) -> None:
        if self.held:
            try:
                if self.path.read_text().strip() == str(os.getpid()):
                    self.path.unlink()
            except OSError:
                pass
            self.held = False
