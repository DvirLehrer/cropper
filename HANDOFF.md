# Handoff — cropper rework, July 2026

Written for whoever picks this up next. `CLAUDE.md` documents how the pipeline
works; this file documents *why we are changing it* and what state the work is in.

## The situation

The company (IL"M / stamscanner.co.il) runs a STaM proofreading app. Its error
detection is good; its letter recognition from photographs is brittle — it
rejects images that are too sharp, slightly dark, motion-blurred, or carry a
small stain. Rather than fix the recogniser, they contracted Dvir to build a
cropper that pre-processes the photo into something the app can digest.

So the cropper is not judged on "is the crop visually correct". It is judged on
**whether the downstream engine reads the parchment correctly afterwards.**

Contract deliverables (נספח א'), which map 1:1 onto the benchmark folders:

1. Crop correctly on any background / lighting — ≥80% of a preselected set
2. Perspective correction for unaligned corners, preserving aspect ratio
3. Detect a vertical parchment and rotate 90° CCW
4. Exclude foreign objects — magnets, tape, table background
5. Ignore ruled lines, frames and engravings
6. Handle over-high-resolution images where parchment grain reads as strokes
7. Integration: no new exceptions, ≤10% processing-time increase vs the prior version

## Where we stand

The company ran an independent comparison on 21 Jul 2026 across four cropper
versions, 144 images, scored against the real production engine. This repo is
their "cropper #3":

| Version | Success | Their error rate | Time |
|---|---|---|---|
| raw, no crop | 49.3% | 47.9% | — |
| #1 first delivery | 79.2% | 35.1% | 6.94 s |
| #2 repo tip | 84.7% | 37.3% | 6.83 s |
| **#3 this repo** | **84.7%** | **42.5%** | **2.27 s** |
| #4 in production | 81.2% | 39.1% | 3.51 s |

We are the fastest in all six categories — 3× faster than #1, and the 10% clause
in criterion 7 is measured *relative to the prior version*, so speed is not a
risk. We are the most accurate in perspective, objects and drawings. We are the
worst of all four versions in exactly two places:

- **roughness** — 47.5% error rate vs 27.7% for #2
- **rotate** — 105.2%, i.e. more errors than there are words. Even the nominal
  successes are unusable.

Those two are the job.

## Where this stands — measured, not estimated

Everything below was measured with `tools/bench.py` against the company's own
Stam-OCR engine, on the frozen 157-image benchmark (or the 43-image dev set
where noted). The best configuration so far is the `region` run.

**The two results that need no interpretation:**

| | untouched photo | after our crop |
|---|---|---|
| the engine returns a result at all | 83/157 — 52.9% | **137/157 — 87.3%** |
| of the known reference text, how much it read | 40.5% | **77.5%** |

Put plainly: the engine fails outright on 47% of real customer photos and on 13%
after our crop. It rescued 57 images and broke 3. Neither number depends on any
definition anyone could argue about.

Per challenge, text recovered (dev set):

| | ours |
|---|---|
| general | 99.3% |
| perspective | 90.6% |
| drawings | 90.7% |
| objects | 93.5% |
| cropper | 84.3% |
| **roughness** | **51.5%** |
| **rotate** | **51.3%** |

Roughness and rotate are the whole remaining problem.

## What was tried, and what the numbers said

Kept:

- **HEIC decoding** — 17 of 157 images no cropper could open, including theirs.
- **One decoded array to both YOLO and Vision** — removed an EXIF frame
  mismatch and two redundant decodes per image.
- **Deskew by character baseline angle**, replacing a rule that rotated any
  portrait page. Residual angle in the output is now a median 0.5°, 3% above 2°.
- **Text region instead of a convex hull** (`_text_region`). The largest single
  win: 73.8% -> 77.5% text recovered. A convex hull lets one stray character box
  drag the outline out and swallow the whole triangle in between — the wedges of
  table and wall visible in the rotate crops.

Rejected, with the measurement that killed each:

- **Ruled-line suppression** (`deruling.py`, kept behind `DERULE=0`). Doubles
  the drawings score where the parchment really is ruled, but fires on unruled
  pages and bleaches the letters: 32.5% -> 31.8% under-20 ungated, 30.6% gated.
  The ceiling with a perfect gate is 35.7%, so the remaining headroom is small.
- **Leashing the model polygon to the recognised text** (`MODEL_LEASH_CHARS=0`).
  Looked obvious — a quarter of the median rotate crop is area with no *detected*
  ink — and cost perspective 93.8% -> 70.5%. That area holds text too faint for a
  threshold to call ink. The measurement was wrong, not the pipeline.
- **A wider text reach**, 0.6 -> 1.2 characters: 77.5% -> 73.6%.

Never isolated, still unknown: the crop margin (`MARGIN_CHARS`, at 0) and
background flattening (`FLATTEN_BG`, on since before the benchmark existed).

## How to run an experiment without fooling yourself

Every tunable is overridable from the environment and every run records its
settings, because twice in one session a run meant to test a new value silently
measured the old one and the near-identical numbers were nearly taken as
evidence:

```bash
TEXT_REACH_CHARS=1.2 python3 tools/bench.py crop --run reach12 --set ../benchmark/dev_set.txt
.venv-scoring/bin/python tools/bench.py score --run reach12
python3 tools/bench.py report --run reach12 --vs region
```

`report --vs` prints exactly which settings differed, and says so loudly when
none did. Change one thing per run; the day this file describes lost two
experiments to bundled changes.

Search on `../benchmark/dev_set.txt` — 43 images, stratified by challenge *and*
by outcome, fixed on disk. Confirm the winner on all 157 before believing it.

## What actually counts as success

Told to us by the company directly, and **not written anywhere in their report**:

> Above roughly **20 errors** on an image, the engine is no longer finding real
> scribal faults — it has failed to read the picture. A genuine scan usually
> shows a handful. Zero errors is not the goal either; it means almost nothing
> was read.

So the target is the **share of images that come back under 20 errors**, not an
average error rate. Their report measured something else entirely: "success" as
"the crop yielded readable text at all" (their words), plus errors-over-words
averaged only across successes. Those two definitions grade the same images very
differently — 84.7% by theirs, plausibly around 30% under the 20-error rule,
which is likely where the "you're at 30%, nowhere near 80%" remark came from.

`bench.py report` therefore leads with `<20` and keeps recovery beside it as a
guard: a crop that discards half the parchment also reports few errors, and must
not be allowed to look like a win.

Worth requesting from the company: `שגיאות לפי תמונה ולפי קרופר.csv`, which
already holds the per-image error counts for all four croppers. It would let us
compute the under-20 rate for every version without running anything.

## Settled: the Google Vision dependency is accepted

The report's main reservation about this cropper was its reliance on Google
Cloud Vision — cost, availability, third-party risk — and it hedged toward a
hybrid or toward cropper #2 if the cloud call were unacceptable. The company has
since confirmed the dependency is fine. That removes the only blocker no amount
of code could fix, and makes this repo the candidate to finish.

## The blockers, as of today

| # | Blocker | Status |
|---|---|---|
| 1 | Rotate — 105.2% error rate, worse than useless | mechanism replaced, unmeasured |
| 2 | Roughness — 47.5%, worst of all four versions | untouched |
| 3 | `NO_DETECTION` on 10 images, a failure mode the Tesseract-based versions cannot have | our sweep returned 157/157, unverified |
| 4 | Basic crop clips letters — 26% errors on the crop folder, worst of four, despite 100% "success" | diagnosed, not fixed |
| ~~5~~ | ~~Google Vision dependency~~ | accepted by the company |

Note that (4) is ours, not theirs: their report never separated it out, because
their success metric could not see it.

## Design principle: detect, then treat

Every stage should decide per image whether it is needed, and do nothing when it
is not. A pipeline that rotates what is already straight, or blurs what is
already clean, loses on the images it should have left alone — and those are the
majority.

Partly honoured already: deskew only fires above `DESKEW_MIN_DEG`. Not honoured
in background flattening, which triggers on crop size and aspect ratio rather
than on whether the parchment is actually grainy — so a large clean scan gets
blurred for nothing while a small rough one is left untouched. That mismatch is
a prime suspect for the roughness category and should be fixed by measuring
grain (e.g. high-frequency energy in the background mask) and gating on it.

## The rotation problem is not 90°

Appendix A criterion 3 says "detect a vertical parchment and rotate 90° left",
but the benchmark images say otherwise. `02_rotate/IMG_1221` is a mezuza
photographed at roughly 25°; `IMG_1219` is a tefillin strip at an arbitrary
angle *and* needing 90°. The real task is **arbitrary-angle correction**, with
the 90° case as one instance of it.

This also means rotation and perspective are one problem, not two, and should be
solved together: the YOLO polygon is already computed, so `cv2.minAreaRect` on it
yields the angle for free, and fitting it to a quadrilateral (`approxPolyDP`)
plus one `warpPerspective` handles keystone in the same pass. Criterion 2 was
dropped from the second delivery because it was expensive; done this way it is
tens of milliseconds on data we already have.

### Already done, not yet measured

The old `_ocr_is_vertical()` — which rotated whenever the OCR bounding box was
taller than wide, and therefore wrongly rotated every portrait mezuza — has been
replaced with a directed baseline angle over the Vision character boxes, applied
as a single `warpAffine` that rotates and crops together.

Verified independently against the 436 cached `test_output/*_char_boxes.json`:
420 of 436 scans measure within 0.5° of straight, 4 sit at exactly 90°, and
**zero** land near 180°. That last one mattered: had Vision ordered its box
vertices along RTL reading direction, `v0→v1` would point left, the mean angle
would come out ~180°, and every straight page would be flipped upside down. It
does not. The code comment claiming "434 of 436" is slightly off; the figure is
420.

Perspective correction still does not exist.

## Prime suspect: the rotation heuristic (resolved — kept for context)

`crop_stam.py`, in `crop_image()`:

```python
def _ocr_is_vertical() -> bool:
    ...
    return (max(cy) - min(cy)) > (max(cx) - min(cx))

if h / max(w, 1) >= ROTATE_RATIO or _ocr_is_vertical():
    cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
```

This rotates whenever the OCR bounding box is taller than it is wide. That is
true of any portrait parchment — which is most mezuzot — whose lines already run
horizontally. Those get rotated 90° for no reason, and the engine then reads
sideways text.

The correct signal is the direction the text **lines** run, not the shape of the
text block. With Vision character boxes in hand, that is cheap to compute: take
each character's nearest neighbours and look at the dominant direction of the
offsets. Horizontal ⇒ leave alone. Vertical ⇒ rotate.

This has since been confirmed and replaced — see "Already done, not yet
measured" above. Kept here because it explains the 105.2% figure.

## The basic crop clips letters

Easy to miss, because criterion 1 passes on both readings: 100% (35/35) on the
crop folder, 84.7% across all 144. But on that same crop folder our error rate is
26% — the **worst of all four versions** (17.6% / 18.3% / 21.4%). Text is read,
and read badly. That is the signature of a crop that slices characters.

Two concrete causes, both cheap to fix:

1. **No margin at all.** The crop is `cv2.boundingRect(boundary)`, and the OCR
   hull passes exactly through the outermost character corners. For STaM this
   bites harder than for ordinary text: the *tagin* on שעטנז ג"ץ rise above the
   glyph body, and Vision's character boxes are tight to the body, so the crowns
   fall outside the hull and get cut.
2. **`expand_to_blobs()` is never called.** It exists at `crop_with_model.py:122`
   and absorbs ink lying just outside the polygon; `crop_with_model.crop_image`
   uses it. The production path in `crop_stam.py` does not. The safety net
   against clipping is already in the repo and simply is not wired in.

## What was done in this session

**Benchmark rebuilt** — `../benchmark/`, 157 images in six per-challenge folders
plus a legacy set, deduplicated by content hash (55 duplicates removed, including
two identical rotate zips). `manifest.csv` carries dimensions, true format, hash
and every alias each image arrived under. See `../benchmark/README.md`, which
also reconciles our 157 against their 144. Rebuild with
`tools/organize_benchmark.py --root ../benchmark` (idempotent).

**HEIC support** — new `stam_io.py`. 17 benchmark images are HEIC files carrying
a `.jpg` extension: 13 of 28 perspective, 4 of 29 rotate. `cv2.imread` returns
`None` for those, so *every* cropper tested — ours included — never decoded them
at all, and the failure surfaced downstream looking like a detection failure.
Requires `pillow-heif`.

**EXIF frame bug fixed** — `crop_image()` handed the file *path* to Ultralytics,
which ran its own `cv2.imread`, which applies EXIF orientation, while our decode
and Google Vision both ignore it. For an image tagged `orientation=6` the model
polygon came back in a frame rotated 90° from the OCR boxes. Both branches now
receive the same decoded array. Three benchmark images were affected; the change
also removes two redundant decodes per image.

**Benchmark harness** — `tools/bench.py`, three independent stages:

```bash
python3 tools/bench.py crop   --run baseline     # fast: cropping + timing only
python3 tools/bench.py score  --run baseline     # slow: real Stam-OCR engine
python3 tools/bench.py report --run baseline --vs previous
```

Scores are cached by the SHA-256 of each crop's bytes in a shared
`score_cache.jsonl`, so a change that only moves rotate images re-scores only
those. Only the join/report path has been exercised, against synthetic fixtures
— the crop and score stages have never been run for real.

**macOS scoring environment** — `tools/setup_scoring_macos.sh` plus
`tools/requirements-scoring-macos.txt`. Native Apple Silicon, no Docker. Needs
Python 3.11 (TF 2.15 publishes no cp312 wheels) and swaps `tensorflow-cpu` for
`tensorflow-macos` at the same 2.15.1 — do not move to TF 2.16+, where
`tensorflow.keras` becomes Keras 3 and the legacy `.h5` load path changes.

## The metric, and why it is not theirs

Their report scored `errors / words_detected`, averaged over images that
succeeded. Both halves are unstable:

- The denominator is what the OCR *found*. A cropper that cuts away half the
  parchment reports fewer words, therefore fewer errors, and scores better.
- Averaging over successes only grades each version on a different subset. A
  version that recovers a hard image is penalised for its errors; a version that
  fails that image outright is simply not graded on it. Note our word average is
  164.4 against 143.4 for #1 — we recover more text and are charged for it.

Stam-OCR diffs against a **known reference text**: `ShowResults.find_reference_text`
auto-detects the document type (`TEXT_TYPE_AUTO_DETECT = True`) and stores the
matched page in `scan_data.compare_data_module.reference_text`. That gives a
denominator fixed per document type and independent of crop quality.

So `bench.py` reports **recovery** = (letters read − letters flagged) / letters
in the reference text, with a failed crop scoring 0 rather than dropping out of
the average. Their metric is computed alongside for cross-checking.

One detail worth preserving: `bench.py` passes a *hash* as the filename to
`create_scan`, because `ScanData` derives `text_type` from the filename stem — an
image called `mezuza3.jpg` is forced to the mezuza reference while `IMG_1219.jpg`
is auto-detected. Uniform auto-detection keeps every image on the same footing.

## Open items

- **`controller.js` is missing.** It lives in the company's server repo, which is
  not accessible — the GitHub org exposes only `Stam-OCR`. It decides which of
  the 20 error types in `parameters.py:189` are counted in the customer-facing
  report. Without it our absolute numbers will not match theirs, though deltas
  between our own runs remain valid.
- **Ask Moshi for `שגיאות לפי תמונה ולפי קרופר.csv` and `final_summary.json`.**
  He has already produced both. They are the only way to verify our harness
  reproduces his baseline rather than being quietly broken.
- **The "preselected image set" in criterion 1 is undefined.** Freezing the 157
  matters for engineering reasons regardless: without a stable set, a change
  cannot be attributed to the code rather than to the test set.
- **Roughness** has had no analysis yet. The hypothesis in their report is that
  our Otsu threshold plus background fill amplifies grain instead of smoothing
  it, which would make the flattening step in `crop_image()` the place to look.
- **Objects has only 9 images**, so each one is worth 11% — too small to support
  any threshold. Worth asking for more magnet/tape samples.

## How we measure: ourselves against ourselves

Their numbers cannot be reproduced exactly — we lack `controller.js`, our set is
157 rather than 144, and our headline metric is deliberately different. None of
that matters, because the comparison that decides whether a change helped is
**our own code before versus after**, measured with one tool on one frozen set.

Nothing here depends on the company:

```bash
git stash                                     # the code as they tested it
python3 tools/bench.py crop  --run before
python3 tools/bench.py score --run before

git stash pop                                 # with our fixes
python3 tools/bench.py crop  --run after
python3 tools/bench.py score --run after

python3 tools/bench.py report --run after --vs before
```

Their figures matter only for the final conversation with Moshi.

## Looking at the results

`cropper/current/` always holds the latest state and is overwritten on every
run: `index.html` (open it — original and crop side by side, filterable by
challenge, by failure, by rotated; click zooms the full-resolution file),
`crops/` (every current crop at full size) and `_thumbs/`. `bench.py crop`
rebuilds it every 10 images, so it can be opened mid-sweep.

Numbers alone will not catch a crop that is subtly wrong on every image. Look at
the sheet after each run.

## Next steps, in order

1. `./tools/setup_scoring_macos.sh`, then `pip install pillow-heif` in the normal
   environment.
2. Establish before/after as above. This is the first measurement anyone has run
   since the changes; expect the first sweep to surface plumbing problems.
3. Look at `current/index.html` and sanity-check the crops by eye.
4. Then change algorithms, one challenge at a time, re-measuring each time:
   - **crop clipping** — margin plus `expand_to_blobs`. Cheapest fix, worst
     current standing, affects every category.
   - **rotation and perspective together** — `minAreaRect` / `approxPolyDP` on
     the polygon we already have.
   - **roughness** — gate the flattening on measured grain instead of on crop
     size.
   - leave objects, drawings and general backgrounds alone except to confirm
     they have not regressed; we already lead there and the headroom is a few
     percent.
