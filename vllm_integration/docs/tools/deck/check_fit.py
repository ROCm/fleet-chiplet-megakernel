#!/usr/bin/env python3
"""Estimate rendered height of every slide in fleet-mk-talk.html.

There is no headless renderer on this box, so "does it fit on one 16:9 page"
is computed rather than looked at. The model is deliberately crude and
deliberately pessimistic: it walks the block-level children of each .slide and
sums an estimated height for each, wrapping text at a measured average advance
width. It will not be exact. It is reliable for the only question that matters
-- is a slide near or over the 630 px box.

    python3 check_fit.py [path]

Slides within ~40 px of the limit are flagged TIGHT; over it, OVER.
"""
import re
import os
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "fleet-mk-talk.html"))
PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
PAGE_H = 630
PAD_V = 44 + 58          # .slide padding top + bottom
CONTENT_W = 1120 - 56 * 2  # .slide padding left/right
CHAR_W = 0.50            # advance width as a fraction of font-size

# (font-size, line-height, margin-top, margin-bottom) per block element,
# read off the deck stylesheet. Top and bottom are kept separate because CSS
# collapses adjacent vertical margins to the larger of the two rather than
# summing them -- treating them as a sum overestimates a dense slide by 10-15%,
# which is enough to report a false OVER.
BLOCK = {
    "h1": (40, 1.15, 0, 10),
    "h2": (29, 1.20, 14, 6),
    "p":  (16, 1.60, 11, 11),
    "li": (16, 1.60, 9, 9),
    "figcaption": (12.5, 1.45, 7, 0),
    "tr": (14.5, 1.50, 0, 0),
}


def inline_margin(style):
    """(top, bottom) from a `margin:` shorthand in an inline style, or None."""
    m = re.search(r"margin:\s*([^;\"]+)", style)
    if not m:
        return None
    parts = [p for p in m.group(1).split() if p]
    px = [0.0 if p in ("0", "auto") else float(p.rstrip("px") or 0) for p in parts]
    if len(px) == 1:
        return px[0], px[0]
    if len(px) == 2:
        return px[0], px[0]
    if len(px) == 3:
        return px[0], px[2]
    return px[0], px[2]


def inline_pad(style, default):
    m = re.search(r"padding:\s*([\d.]+)px", style)
    return float(m.group(1)) if m else default


class Fit(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.slides = []          # (id, height)
        self.h = None             # current accumulated height, None = outside
        self.sid = None
        self.stack = []
        self.buf = ""
        self.cur = None           # current block tag being measured
        self.fs_scale = 1.0       # explicit font-size override via style=""
        self.two = 0              # inside a .two grid -> half width
        self.skip = 0             # inside <svg> / <defs>
        self.depth = 0            # open <div> depth, counted from the slide
        self.two_at = None        # div depth at which the .two grid opened
        self.pre_fs, self.pre_lh = 12.5, 1.5   # current <pre><code> metrics
        self.pending_mb = 0.0     # unresolved bottom margin, for collapsing
        self.mg_override = None   # margin from an inline style="margin:..."

    # ---- helpers ----------------------------------------------------------
    def width(self):
        return CONTENT_W / 2 - 11 if self.two else CONTENT_W

    def flush(self):
        if self.cur is None:
            return
        fs, lh, mt, mb = BLOCK[self.cur]
        fs *= self.fs_scale
        if self.mg_override is not None:
            mt, mb = self.mg_override
        text = re.sub(r"\s+", " ", self.buf).strip()
        per_line = max(1, int(self.width() / (fs * CHAR_W)))
        lines = max(1, -(-len(text) // per_line))
        self.add(lines * fs * lh, mt, mb)
        self.cur, self.buf, self.fs_scale = None, "", 1.0
        self.mg_override = None

    def add(self, px, mt=0.0, mb=0.0):
        """Add a block of height px, collapsing its top margin against the
        previous block's bottom margin."""
        if self.h is None:
            return
        gap = max(self.pending_mb, mt)
        self.h += (gap + px) / 2 if self.two else gap + px
        self.pending_mb = mb

    # ---- parser -----------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")

        if tag == "svg":
            self.skip += 1
            if self.skip == 1 and self.h is not None:
                vb = a.get("viewBox")
                if vb:
                    _, _, w, hh = (float(v) for v in vb.split())
                    self.add(hh * (self.width() / w), 14, 7)   # <figure> margin
            return
        if self.skip:
            return

        if tag == "div" and "slide" in cls.split():
            self.sid, self.h, self.depth = a.get("id", "?"), float(PAD_V), 0
            self.pending_mb = 0.0
            return
        if self.h is None:
            return
        if tag == "div":
            self.depth += 1

        # A .two grid halves the available width; its two columns stack in the
        # estimate, so each contributes half its height. Tracked by the div
        # depth it opened at, since the columns are divs too.
        if tag == "div" and "two" in cls.split() and self.two_at is None:
            self.two, self.two_at = 1, self.depth
        if tag in ("div", "ul", "table", "pre", "figure"):
            self.stack.append(tag)
        mg = inline_margin(a.get("style", ""))
        if tag == "div" and "box" in cls.split():
            pad = inline_pad(a.get("style", ""), 13)
            self.add(pad * 2 + 32, *(mg if mg else (16, 16)))   # padding + .lbl
        if tag == "div" and "stat-row" in cls.split():
            self.add(62, *(mg if mg else (18, 18)))
        if tag == "pre":
            pad = inline_pad(a.get("style", ""), 12)
            self.add(pad * 2 + 2, *(mg if mg else (12, 12)))
        if tag == "table":
            self.add(32, *(mg if mg else (14, 14)))             # header row
        if tag == "hr":
            self.add(1, *(mg if mg else (16, 16)))
        if tag == "code" and self.stack and self.stack[-1] == "pre":
            # Counted line-by-line in handle_data; pick up any font override,
            # since the code slide shrinks its listing to fit.
            self.cur = None
            st = a.get("style", "")
            m = re.search(r"font-size:\s*([\d.]+)px", st)
            self.pre_fs = float(m.group(1)) if m else 12.5
            m = re.search(r"line-height:\s*([\d.]+)", st)
            self.pre_lh = float(m.group(1)) if m else 1.5

        if tag in BLOCK:
            self.flush()
            self.cur = tag
            st = a.get("style", "")
            self.mg_override = inline_margin(st)
            m = re.search(r"font-size:\s*([\d.]+)px", st)
            if m:
                self.fs_scale = float(m.group(1)) / BLOCK[tag][0]
            elif "big" in cls.split():
                self.fs_scale = 23 / BLOCK[tag][0]
            elif "kicker" in cls.split():
                self.fs_scale = 19 / BLOCK[tag][0]
            elif "sub" in cls.split():
                self.fs_scale = 17 / BLOCK[tag][0]
            elif "cite" in cls.split():
                self.fs_scale = 12.5 / BLOCK[tag][0]

    def handle_endtag(self, tag):
        if tag == "svg":
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in BLOCK:
            self.flush()
        if tag in ("div", "ul", "table", "pre", "figure") and self.stack:
            self.stack.pop()
        if tag != "div" or self.h is None:
            return
        if self.depth == 0:                       # the slide's own </div>
            self.slides.append((self.sid, round(self.h)))
            self.sid, self.h, self.two, self.two_at = None, None, 0, None
            return
        if self.two_at == self.depth:
            self.two, self.two_at = 0, None
        self.depth -= 1

    def handle_data(self, data):
        if self.skip or self.h is None:
            return
        if self.stack and self.stack[-1] == "pre":
            self.add(data.count("\n") * self.pre_fs * self.pre_lh)
            return
        if self.cur:
            self.buf += data


f = Fit()
f.feed(open(PATH).read())

worst = 0
for sid, h in f.slides:
    state = "ok   "
    if h > PAGE_H:
        state, worst = "OVER ", max(worst, 1)
    elif h > PAGE_H - 40:
        state = "TIGHT"
    print("  %-5s %-5s %4d px  (budget %d)" % (state, sid, h, PAGE_H))

print("\n%d slides, tallest %d px" % (len(f.slides), max(h for _, h in f.slides)))
sys.exit(worst)
