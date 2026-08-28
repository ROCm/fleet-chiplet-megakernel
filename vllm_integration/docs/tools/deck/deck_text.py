#!/usr/bin/env python3
"""Dump titan-talk.html as plain text, one block per line, slide by slide.

A regex over `<div class="slide">…</div>` truncates at the first nested
`</div>`, which silently hides most of a slide's content -- the reason this
exists as a parser instead. Use it to proof-read the deck without a browser.

    python3 deck_text.py [path]
"""
import os
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "titan-talk.html"))
PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
BLOCK = {"h1", "h2", "p", "li", "figcaption", "tr", "pre"}


class Dump(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.buf = ""
        self.cur = None
        self.depth = None      # open-div depth inside a slide, None = outside
        self.skip = 0          # inside <svg>
        self.in_index = False

    def emit(self):
        text = " ".join(self.buf.split())
        if text:
            self.out.append("  %-11s %s" % (self.cur + ":", text))
        self.cur, self.buf = None, ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "").split()
        if tag == "svg":
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "div" and "index" in cls:
            self.in_index = True
        if tag == "div" and "slide" in cls:
            self.out.append("\n=== %s ===" % a.get("id", "?"))
            self.depth = 0
            return
        if self.in_index or self.depth is None:
            return
        if tag == "div":
            self.depth += 1
        if tag in BLOCK:
            self.emit()
            self.cur = "tag" if False else tag

    def handle_endtag(self, tag):
        if tag == "svg":
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == "div" and self.in_index:
            self.in_index = False
            return
        if tag in BLOCK:
            self.emit()
        if tag != "div" or self.depth is None:
            return
        if self.depth == 0:
            self.depth = None
        else:
            self.depth -= 1

    def handle_data(self, data):
        if self.skip or self.depth is None or self.in_index:
            return
        if self.cur:
            self.buf += data
        elif data.strip():                      # bare text: .tag / .num chrome
            self.out.append("  %-11s %s" % ("·", " ".join(data.split())))


d = Dump()
d.feed(open(PATH).read())
print("\n".join(d.out))
