#!/usr/bin/env python3
"""Anchor-based paragraph editor for titan-architecture.html.

The doc is hand-wrapped at ~95 columns and mixes literal glyphs (x, ->) with
inline <em>/<strong>/<code>, so exact-string matching against a remembered
paragraph almost never works. Instead: locate the unique <p> that contains an
anchor substring, and replace that whole element.

    from edit_para import Doc
    d = Doc('titan-architecture.html')
    d.replace_p('anchor text inside the paragraph', '<p>new html</p>')
    d.drop_p('anchor text')          # delete the paragraph entirely
    d.save()                          # prints the byte delta

Every operation asserts the anchor matched exactly one paragraph, so a stale
anchor is a loud failure rather than a silent no-op.
"""
import re
import sys


class Doc:
    def __init__(self, path):
        self.path = path
        self.text = open(path).read()
        self.orig = len(self.text)
        self.log = []

    @staticmethod
    def _plain(html):
        """Tag-stripped, whitespace-collapsed rendering of a fragment."""
        return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', html)).strip()

    def _paragraphs(self):
        """Yield (start, end) for every <p>...</p> in the document."""
        for m in re.finditer(r'<p>.*?</p>', self.text, re.S):
            yield m.start(), m.end()

    def _find_p(self, anchor):
        """Return (start, end) of the <p>...</p> uniquely matching anchor.

        The anchor is matched against the paragraph's *tag-stripped* text with
        whitespace collapsed. Both are necessary: the doc is hand-wrapped at
        ~95 columns, so any anchor longer than a few words crosses a line
        break, and nearly every paragraph carries inline <code>/<em>/<strong>
        that a literal anchor would have to reproduce exactly.
        """
        want = self._plain(anchor)
        hits = [(a, b) for a, b in self._paragraphs()
                if want in self._plain(self.text[a:b])]
        if len(hits) != 1:
            raise SystemExit(f"anchor matched {len(hits)} paragraphs: {anchor!r}")
        return hits[0]

    def replace_p(self, anchor, new_html):
        a, b = self._find_p(anchor)
        old_len = b - a
        self.text = self.text[:a] + new_html + self.text[b:]
        self.log.append((anchor[:44], len(new_html) - old_len))
        return self

    def drop_p(self, anchor):
        a, b = self._find_p(anchor)
        # swallow the trailing newline so we do not leave a blank gap
        while b < len(self.text) and self.text[b] == '\n':
            b += 1
        self.log.append((anchor[:44], -(b - a)))
        self.text = self.text[:a] + self.text[b:]
        return self

    def save(self):
        open(self.path, 'w').write(self.text)
        for anchor, delta in self.log:
            print(f"{delta:+7d}  {anchor}")
        print(f"\n{self.orig} -> {len(self.text)} "
              f"({len(self.text) - self.orig:+d} bytes)")


if __name__ == '__main__':
    print(__doc__)
    sys.exit(0)
