#!/usr/bin/env python3
"""
fix_notebook.py

Fixes the classic Colab -> GitHub notebook rendering error:
    "Invalid Notebook: the 'state' key is missing from 'metadata.widgets'."

Usage:
    python fix_notebook.py path/to/notebook.ipynb
    python fix_notebook.py notebooks/*.ipynb          # fix multiple at once

It edits the file(s) IN PLACE (safe — only removes/repairs the
metadata.widgets block, never touches your code or outputs).
"""

import json
import sys
import glob


def fix_notebook(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    widgets = nb.get("metadata", {}).get("widgets")

    if widgets is not None:
        # If it's missing 'state' inside the widget-state block, or is
        # otherwise malformed, just drop it. Widget UI doesn't render on
        # GitHub anyway, so this never affects your visible code/output.
        needs_removal = True
        wjson = widgets.get("application/vnd.jupyter.widget-state+json")
        if isinstance(wjson, dict) and "state" in wjson:
            needs_removal = False  # already valid, leave it alone

        if needs_removal:
            del nb["metadata"]["widgets"]
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
        print(f"Fixed: {path}")
    else:
        print(f"OK (no change needed): {path}")

    return changed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_notebook.py <notebook.ipynb> [more.ipynb ...]")
        sys.exit(1)

    paths = []
    for pattern in sys.argv[1:]:
        paths.extend(glob.glob(pattern))

    if not paths:
        print("No matching files found.")
        sys.exit(1)

    any_changed = False
    for p in paths:
        try:
            any_changed = fix_notebook(p) or any_changed
        except Exception as e:
            print(f"Error processing {p}: {e}")

    sys.exit(0)