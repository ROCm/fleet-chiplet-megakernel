#!/usr/bin/env python3
"""Validation passes for titan-architecture.html.

There is no headless renderer on this box, so "does it look right" is checked
numerically instead. Four passes:

  1. SVG well-formedness  -- every inline <svg> parses as XML.
  2. SVG geometry         -- rects inside the viewBox, and estimated <text>
                             extents inside it too. Elements carrying a
                             transform are skipped: rotated labels report as
                             out-of-bounds and are false positives.
  3. HTML tag nesting     -- an HTMLParser pass over the whole file, since a
                             mis-nested </div> shows up in neither SVG check.
  4. Reference integrity  -- every href="#id" resolves, figure captions are
                             numbered 1..N in document order, and no <defs>
                             marker id is defined twice (inline SVGs share one
                             DOM, so a duplicate id silently wins over the rest).

Also reports the prose budget: paragraph count and total tag-stripped chars.

    python3 tools/validate_doc.py [path]
"""
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
        'link', 'meta', 'param', 'source', 'track', 'wbr'}
CHAR_W = 0.52          # empirical advance width as a fraction of font-size


def check_svg(text, fail):
    svgs = re.findall(r'<svg\b.*?</svg>', text, re.S)
    print(f"[1] well-formedness: {len(svgs)} inline <svg>")
    roots = []
    for i, svg in enumerate(svgs, 1):
        try:
            roots.append(ET.fromstring(svg))
        except ET.ParseError as e:
            fail(f"    svg #{i} does not parse: {e}")
            roots.append(None)

    print("[2] geometry")
    for i, root in enumerate(roots, 1):
        if root is None:
            continue
        vb = root.get('viewBox')
        if not vb:
            # The shared <defs> block is width=0 height=0 and deliberately has
            # no viewBox -- it renders nothing and only supplies marker ids.
            if root.get('width') == '0':
                continue
            fail(f"    svg #{i} has no viewBox")
            continue
        x0, y0, w, h = (float(v) for v in vb.split())

        # font-size and text-anchor are inherited, and in this doc they are
        # almost always set on an enclosing <g> rather than on the <text>.
        # Reading only the element's own attributes reports every centred
        # label as a left-anchored overrun.
        parent = {c: el for el in root.iter() for c in el}

        def inherited(el, attr, default):
            node = el
            while node is not None:
                v = node.get(attr)
                if v is not None:
                    return v
                node = parent.get(node)
            return default

        def transformed(el):
            node = el
            while node is not None:
                if node.get('transform'):
                    return True            # rotated -> false positives
                node = parent.get(node)
            return False

        for el in root.iter():
            if transformed(el):
                continue
            tag = el.tag.rsplit('}', 1)[-1]
            if tag == 'rect':
                try:
                    rx, ry = float(el.get('x', 0)), float(el.get('y', 0))
                    rw, rh = float(el.get('width', 0)), float(el.get('height', 0))
                except ValueError:
                    continue
                if rx < x0 - .5 or ry < y0 - .5 or rx + rw > x0 + w + .5 \
                        or ry + rh > y0 + h + .5:
                    fail(f"    svg #{i} rect ({rx},{ry},{rw},{rh}) "
                         f"outside viewBox {vb}")
            elif tag == 'text':
                # Collapse whitespace: the source wraps long labels across
                # lines, and the indentation is not rendered as glyphs.
                s = re.sub(r'\s+', ' ', ''.join(el.itertext())).strip()
                if not s:
                    continue
                try:
                    fs = float(inherited(el, 'font-size', 12))
                    tx = float(el.get('x', 0))
                except ValueError:
                    continue
                ext = len(s) * fs * CHAR_W
                anchor = inherited(el, 'text-anchor', 'start')
                lo = tx - ext / 2 if anchor == 'middle' else \
                     tx - ext if anchor == 'end' else tx
                if lo < x0 - 2 or lo + ext > x0 + w + 2:
                    fail(f"    svg #{i} text overruns "
                         f"[{lo:.0f}..{lo + ext:.0f}] vs viewBox {vb}: {s[:50]!r}")


def check_nesting(text, fail):
    print("[3] tag nesting")

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack = []

        def handle_starttag(self, tag, attrs):
            if tag not in VOID:
                self.stack.append((tag, self.getpos()[0]))

        def handle_endtag(self, tag):
            if tag in VOID:
                return
            if not self.stack:
                fail(f"    </{tag}> with empty stack at line {self.getpos()[0]}")
            elif self.stack[-1][0] != tag:
                open_tag, line = self.stack[-1]
                fail(f"    </{tag}> at line {self.getpos()[0]} closes "
                     f"<{open_tag}> opened at line {line}")
                self.stack.pop()
            else:
                self.stack.pop()

    p = P()
    p.feed(text)
    for tag, line in p.stack:
        fail(f"    <{tag}> opened at line {line} never closed")


def check_refs(text, fail):
    print("[4] references")
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    for href in set(re.findall(r'href="#([^"]+)"', text)):
        if href not in ids:
            fail(f"    href=#{href} resolves to nothing")

    figs = re.findall(r'<b>Figure\s+(\d+)', text)
    want = [str(i) for i in range(1, len(figs) + 1)]
    if figs != want:
        fail(f"    figure captions out of order: {figs} != {want}")
    else:
        print(f"    figures 1..{len(figs)} in document order")

    for mid in re.findall(r'<marker[^>]*\bid="([^"]+)"', text):
        if len(re.findall(rf'<marker[^>]*\bid="{re.escape(mid)}"', text)) > 1:
            fail(f"    marker id={mid} defined more than once")

    for ref in set(re.findall(r'§(\d+)', text)):
        if not re.search(rf'<h2[^>]*>{ref}\.', text):
            fail(f"    cross-reference §{ref} has no matching <h2>")


def prose_budget(text):
    paras = re.findall(r'<p>.*?</p>', text, re.S)
    total = sum(len(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', p)).strip())
                for p in paras)
    print(f"\nprose: {len(paras)} paragraphs, {total} chars "
          f"({total / max(len(paras), 1):.0f} avg)")
    # len() counts characters; the file has multi-byte glyphs (× → §) so the
    # on-disk byte count is larger. `wc -c` and this number will not agree.
    print(f"file:  {len(text)} chars")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'titan-architecture.html'
    text = open(path).read()
    errors = []

    def fail(msg):
        errors.append(msg)
        print(msg)

    check_svg(text, fail)
    check_nesting(text, fail)
    check_refs(text, fail)
    prose_budget(text)

    print(f"\n{'FAIL: ' + str(len(errors)) + ' problem(s)' if errors else 'PASS'}")
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
