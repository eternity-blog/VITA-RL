#!/usr/bin/env python3
"""Verify every relative link in the repo's Markdown files.

Checks two link styles:
  1. Plain relative links (./x, ../x, path/to/x.md#anchor) -- the target
     must exist on disk, resolved from the linking file's directory.
  2. GitHub-web links of the form ../../blob/main/<path> or
     .../tree/main/<path> (used by docs/00-background/CODEMAP.md) -- the
     '../' prefix depth must match the file's location exactly, and
     <path> must exist in the repo.

Run after any doc edit or file move:  python tools/check_doc_links.py
Exits non-zero if anything is broken, so it can gate a commit or CI job.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
SKIP_DIRS = ("VLMEvalKit/", "web_demo/", "outputs/")  # vendored / generated


def md_files():
    for f in sorted(glob.glob("**/*.md", recursive=True)):
        if not f.startswith(SKIP_DIRS):
            yield f


def main():
    os.chdir(ROOT)
    bad, checked = [], 0
    for f in md_files():
        d = os.path.dirname(f)
        depth = len(d.split("/")) if d else 0
        for m in LINK_RE.finditer(open(f, encoding="utf-8").read()):
            target = m.group(1).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            gm = re.match(r"(?:\.\./)*((?:blob|tree)/main/(.+))", target)
            if gm:
                if not target.startswith("../" * (depth + 2)):
                    bad.append((f, m.group(1), "github-link wrong ../ depth"))
                elif not os.path.exists(gm.group(2)):
                    bad.append((f, m.group(1), "github-link target missing"))
                continue
            if not os.path.exists(os.path.normpath(os.path.join(d, target))):
                bad.append((f, m.group(1), "missing"))
    print(f"checked {checked} relative links in tracked Markdown files")
    for f, link, why in bad:
        print(f"  BROKEN [{why}] {f}: {link}")
    if bad:
        print(f"FAILED: {len(bad)} broken links")
        return 1
    print("OK: zero broken links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
