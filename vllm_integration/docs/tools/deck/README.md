# Building `titan-talk.html`

The 16-slide researcher deck is **generated**. Do not hand-edit
`docs/titan-talk.html` — the next build overwrites it. Edit `slides.py`.

```bash
cd docs/tools/deck
python3 build.py                          # slides.py + head.py + doc figures -> titan-talk.html
python3 check_fit.py                      # does every slide fit on one 1120x630 page?
python3 ../validate_doc.py ../../titan-talk.html
python3 deck_text.py | less               # read the whole deck as plain text
```

Run all four after any edit. `check_fit.py` exits non-zero on an overflowing
slide and `validate_doc.py` on a malformed one, so the three-command sequence is
chainable with `&&`.

| file | what it holds |
|---|---|
| `slides.py` | `SLIDES`: 16 `(tag, html)` pairs. `{FIGn}` splices in figure *n* of the architecture doc. **This is the only file with content in it.** |
| `head.py` | `HEAD`: the standalone `<style>` and the shared SVG `<defs>`. |
| `figures.py` | Extracts the figures out of `titan-architecture.html`. Run it alone to list them. |
| `build.py` | Assembles the three into `docs/titan-talk.html`. |
| `check_fit.py` | Estimates rendered slide height. There is no headless renderer on this box. |
| `deck_text.py` | Tag-stripped dump for proof-reading. |

## Things that will bite you

- **Figures are shared with the document, not copied.** `figures.py` reads them
  out of `titan-architecture.html` at build time, so the two can never drift —
  but a figure edit in the doc silently changes the deck, and figures are
  numbered **positionally**, so inserting one in the doc repoints every
  `{FIGn}` above it. `build.py` prints which figures it used; check that list.
- **A figure that refers to a document section breaks the deck.** A slide has no
  `§3`. `build.py` rewrites those in `CAP_OVERRIDE` (captions) and `SVG_SUB`
  (labels inside the drawing) rather than editing the doc, which is correct on
  its own terms. Both assert their anchor exists, so a doc edit fails the build
  loudly instead of shipping a dangling reference.
- **`check_fit.py` is an estimate**, deliberately pessimistic, accurate to
  roughly ±5%. Treat TIGHT as "look at it in a browser", not as a failure.
- **Self-contained is a hard constraint**: no external CSS, no image files, no
  JS. Print to PDF at 1120&times;630; the print stylesheet hides the deck index
  and puts one slide per page.
