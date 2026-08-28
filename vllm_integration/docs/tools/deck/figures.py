#!/usr/bin/env python3
"""Extract the inline SVG figures from titan-architecture.html.

The deck reuses the document's figures verbatim rather than keeping a second
copy of them, so that editing a figure in the doc updates the deck on the next
build and the two can never drift.

Figures are numbered positionally (1..N in document order), not by parsing the
"Figure N." text in the caption -- the captions are the thing most likely to be
edited, and a renumbering there should not silently repoint a slide at the
wrong drawing.

    python3 figures.py            # list what is available
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.normpath(os.path.join(HERE, "..", "..", "titan-architecture.html"))

FIGURE = re.compile(r"<figure>(.*?)</figure>", re.S)
PARTS = re.compile(r"(<svg\b.*?</svg>)\s*<figcaption>(.*?)</figcaption>", re.S)


def load(path=DOC):
    """{n: {'n', 'svg', 'cap'}} for every <figure> in the document."""
    html = open(path).read()
    figs = {}
    for i, block in enumerate(FIGURE.findall(html), 1):
        m = PARTS.search(block)
        assert m, "figure %d in %s: expected <svg>…</svg><figcaption>" % (i, path)
        figs[i] = {"n": i, "svg": m.group(1).strip(), "cap": m.group(2).strip()}
    assert figs, "no figures found in %s" % path
    return figs


if __name__ == "__main__":
    for n, r in sorted(load().items()):
        vb = re.search(r'viewBox="([^"]+)"', r["svg"])
        cap = re.sub(r"<[^>]+>", "", r["cap"])
        print("%2d  viewBox %-18s  %s" % (n, vb.group(1) if vb else "?", cap[:78]))
