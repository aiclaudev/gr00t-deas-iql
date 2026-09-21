#!/usr/bin/env python3
"""Turn off robosuite's file logger, which writes to a hardcoded shared path.

robosuite 1.4.0 ships macros.py with FILE_LOGGING_LEVEL = "DEBUG", and its
DefaultLogger then attaches a handler to the literal path /tmp/robosuite.log.
On a shared machine that file usually belongs to another user, so merely
importing robosuite raises PermissionError. robosuite's own escape hatch is
macros_private.py, which __init__.py imports ahead of the defaults; this writes
one with file logging disabled.
"""
import pathlib
import sysconfig

package = pathlib.Path(sysconfig.get_path("purelib")) / "robosuite"
defaults = package / "macros.py"
if not defaults.is_file():
    raise SystemExit(f"robosuite not found at {package}")

lines = []
for raw in defaults.read_text().splitlines():
    if raw.startswith("FILE_LOGGING_LEVEL"):
        raw = "FILE_LOGGING_LEVEL = None  # /tmp/robosuite.log is not writable on a shared host"
    lines.append(raw)

private = package / "macros_private.py"
private.write_text("\n".join(lines) + "\n")
print(f"wrote {private}")
