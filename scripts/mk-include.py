#!/usr/bin/env python3

import sys
import re
from pathlib import Path

fn, pf = map(Path, sys.argv[1:])

hd, tl, bl, em = 0, 0, None, None
with open(fn) as fd:
    for ln in fd:
        tl += 1
        m = re.match(r"\s*#\s*\(([0-9]+)\).*", ln)
        if m:
            if bl:
                print(
                    f"""## Block {bl}

.. literalinclude:: {pf}/{fn.name}
   :language: python
   :lines: {hd}-{em}
"""
                )
            hd = tl
            bl = m.group(1)
        if ln.strip():
            em = tl

if bl:
    print(
        f"""## Block {bl}

.. literalinclude:: {pf}/{fn.name}
   :language: python
   :line: {hd}-{tl}
"""
    )
