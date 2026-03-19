"""Shared utilities for provenance tracing modules."""

import difflib
import os
import re


def natural_keys(text: str) -> list:
    """Sort key that handles embedded numbers naturally (1, 2, ... 10)."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def get_history_files(root_dir: str) -> list[dict]:
    """Scan a directory tree for .mlir files, sorted by natural filename order."""
    files = []
    if not os.path.exists(root_dir):
        return []

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".mlir"):
                files.append(
                    {
                        "path": os.path.join(dirpath, f),
                        "name": f,
                        "rel_dir": os.path.basename(dirpath),
                    }
                )
    files.sort(key=lambda x: natural_keys(x["name"]))
    return files


def smart_collapse(prev_text: str, curr_text: str) -> str:
    """Diff two text blocks, collapsing unchanged regions > 6 lines."""
    if not prev_text:
        return curr_text

    prev_lines = prev_text.splitlines()
    curr_lines = curr_text.splitlines()

    matcher = difflib.SequenceMatcher(None, prev_lines, curr_lines)
    output = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            block_len = j2 - j1
            if block_len < 6:
                output.extend(curr_lines[j1:j2])
            else:
                output.extend(curr_lines[j1 : j1 + 2])
                skipped = block_len - 4
                if skipped > 0:
                    output.append(
                        f"    ... [collapsed {skipped} unchanged lines] ..."
                    )
                output.extend(curr_lines[j2 - 2 : j2])
        else:
            output.extend(curr_lines[j1:j2])

    return "\n".join(output)
