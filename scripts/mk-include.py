#!/usr/bin/env python3

# Use this script for convenience when making tutorials
# for readthedocs/sphinx.
#
# Run on a tutorial python file like this
#
# ./mk-include tutorial.py prefix
#
# the script will extract all comments from the
# tutorial script starting with
#
# # (N)
#
# where N is whole number. For each such comment
# it will print out a literal include block for use
# in a sphinx tutorial .rst file, eg
#
# .. literalinclude:: ../../python/example/single_cell_detailed.py
#   :language: python
#   :lines: 98-102
#
# The line numbers are chosen such they start at the
# comment '# (N)' and end just before the next such
# comment (or the end of file).
#
# The prefix argument is added to the basename like this
#
# ./mk-include path/to/tutorial.py prefix/of/docs
#
# gives blocks like this
#
# .. literalinclude:: prefix/of/docs/tutorial.py

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
