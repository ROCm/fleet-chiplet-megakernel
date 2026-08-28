#!/usr/bin/env python3
"""Assemble docs/titan-talk.html from head.py + slides.py + the doc's figures.

    python3 build.py && python3 check_fit.py && \
      python3 ../validate_doc.py ../../titan-talk.html

Reusable: edit slides.py and re-run. Figures are spliced verbatim out of
titan-architecture.html by figures.py, so the deck and the document can never
drift; their captions lose the doc's "Figure N." prefix, since a slide has one
figure and does not need a number.
"""
import re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from head import HEAD                    # noqa: E402
from slides import SLIDES                # noqa: E402
from figures import load as load_figures  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "titan-talk.html"))
FIGS = load_figures()

# The doc's captions open with "<b>Figure N.</b> ". On a slide the figure is
# the only one present, so the number is noise -- strip it, keep the sentence.
CAPNUM = re.compile(r"^\s*<b>Figure\s+\d+\.</b>\s*")

# Captions that refer to somewhere else in the document ("the four rejection
# asserts in §3") have no referent in a deck. Rewritten here rather than in the
# doc, so the doc stays correct on its own terms.
CAP_OVERRIDE = {
    11: ("The economics of the split. A model that fits the existing tile shapes costs a YAML "
         "file, because the library already holds every kernel it needs. A model that does not "
         "costs weeks of device-function work — but that work is written once into the shared "
         "library and every later model inherits it. The generator's four rejection asserts each "
         "name a file and a line: they mark exactly where the green bar ends."),
}


# Same problem one level down: figure 5's SVG labels a gate with the document
# section that introduces it. Substring, not regex -- it must fail loudly if
# the figure changes, and it does, via the §-check in validate_doc.py.
SVG_SUB = {
    5: [("instant · no build · §3 step 2", "instant · no build · generation time")],
}


def figure(n: str) -> str:
    n = int(n)
    r = FIGS[n]
    cap = CAP_OVERRIDE.get(n) or CAPNUM.sub("", r["cap"]).strip()
    svg = r["svg"]
    for old, new in SVG_SUB.get(n, []):
        assert old in svg, "figure %d: %r not found" % (n, old)
        svg = svg.replace(old, new)
    return ('<figure>\n%s\n<figcaption>%s</figcaption>\n</figure>' % (svg, cap))


FIGREF = re.compile(r"\{FIG(\d+)\}")

used, out = [], [HEAD]
for i, (tag, body) in enumerate(SLIDES, 1):
    used += [int(m) for m in FIGREF.findall(body)]
    body = FIGREF.sub(lambda m: figure(m.group(1)), body)
    tag_html = '<div class="tag">%s</div>\n' % tag if tag else ""
    out.append('<div class="slide" id="s%d">\n%s%s\n<div class="num">%d / %d</div>\n</div>\n'
               % (i, tag_html, body.strip(), i, len(SLIDES)))

# Deck contents. Screen only -- hidden by the print stylesheet.
def title_of(body: str) -> str:
    m = re.search(r"<h[12][^>]*>(.*?)</h[12]>", body, re.S)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else "Title"


idx = ['<div class="index"><h2 style="margin-top:0">Titan — deck contents</h2>\n<ol>']
idx += ['<li><a href="#s%d">%s</a></li>' % (i, title_of(b))
        for i, (_, b) in enumerate(SLIDES, 1)]
idx += ['</ol>\n<p class="cite">%d slides, 16:9. Print to PDF at 1120×630. '
        'Full architecture document: <a href="titan-architecture.html">'
        'titan-architecture.html</a>.</p></div>\n' % len(SLIDES)]

html = out[0] + "".join(idx) + "".join(out[1:]) + "</body>\n</html>\n"
open(OUT, "w").write(html)

print("wrote %s  %d chars  %d slides" % (OUT, len(html), len(SLIDES)))
print("figures used:", sorted(set(used)))
print("figures unused:", sorted(set(FIGS) - set(used)))
