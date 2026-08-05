#!/usr/bin/env python3
"""Undo the keystone of a page photographed at an angle, using its own writing.

The deskew step in `crop_stam` rotates the text upright, which is all a rotation
can do. A sheet held at an angle to the lens is a different distortion: the far
edge is smaller than the near one, the text lines are no longer parallel, and no
rotation of any angle straightens them.

Why the text and not the sheet
------------------------------
The obvious route is to find the four corners of the parchment and map them to a
rectangle. It fails on this material. Half the benchmark is a strip against a
wooden table, a sheet under plastic, a mezuza with one corner curled — the
outline is unreliable exactly when the photograph is bad, which is when the
correction is needed.

The writing is the more dependable signal, and it is already measured. Google
Vision has returned a quad for every character, and in a keystoned photograph
the text lines, parallel on the parchment, converge to a vanishing point.

That point alone leaves one degree of freedom. Pinning it from the left-hand
margin was the first attempt and it failed: STaM justifies the right margin and
lets the left fall where the words end, so on these pages the right sits within
12-32 px of a straight line and the left within 66-132. The fit that came out of
those two moved a corner by 49-65% of the image and was refused, correctly, by
the guard below.

What pins it instead is the ruling. The lines are scored into the skin with a
stylus before a word is written, so on the parchment they are parallel *and
evenly spaced*; in the photograph they crowd together as they recede, and the
rate at which they crowd fixes the horizon exactly. Twenty ruled lines rather
than two ragged margins. The column direction is then taken from the justified
margin alone, measured after the projective part is undone.

The cost is one `warpPerspective`, 9 ms on a 3 MP image, against the 575 ms the
Vision call already costs.

The mirror
----------
A vanishing point is a direction without a sign: the point at which the lines
meet is the same whether the text runs left or right. Taken as returned, the
basis can come out left-handed, and the homography then rectifies the page into
its own mirror image — every letter reversed, perfectly straight. It looks
obviously wrong to a person and is invisible to any metric we have, since
mirrored text simply scores zero and joins every other kind of failure. The
directions are therefore forced to point rightwards and downwards, and a
negative determinant aborts the correction rather than trusting it.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


ENABLED = _flag("RECTIFY", "1")

# Degrees of convergence between the first text line and the last, below which
# the page is left alone. Correcting a flat page can only resample it for
# nothing, and `mezuzah1` is the warning: measured at 2.82° by the broken
# grouping it was rectified and went from 49 errors to 60, when its true figure
# is 0.97° and it needed no correction at all.
MIN_FAN = float(os.environ.get("RECTIFY_MIN_FAN", "2.5"))

# Fewer lines than this and the vanishing point is a guess.
MIN_LINES = int(os.environ.get("RECTIFY_MIN_LINES", "4"))
MIN_CHARS_PER_LINE = int(os.environ.get("RECTIFY_MIN_CHARS", "6"))

# A correction that moves a corner by more than this fraction of the image is
# not a keystone, it is a failed fit.
MAX_SHIFT = float(os.environ.get("RECTIFY_MAX_SHIFT", "0.35"))


def _centres(boxes: list) -> list[tuple[float, float]]:
    out = []
    for b in boxes:
        v = b.get("vertices") or []
        if len(v) < 4:
            continue
        xs = [p.get("x", 0) for p in v]
        ys = [p.get("y", 0) for p in v]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys),
                    max(ys) - min(ys)))
    return out


def group_lines(boxes: list) -> list[list[tuple[float, float, float]]]:
    """Chain characters into text lines by following each line along itself.

    Two simpler rules were tried first and both failed on exactly the images
    this module exists for.

    Sorting by height alone: a line on a keystoned page descends further across
    the sheet than one line is tall, so neighbouring lines overlap in y and
    merge. On `mezuzah1` that gave nine groups for twenty lines, two of them
    holding 268 and 248 characters.

    Linking each character to whatever is near it: the links are transitive, and
    where the end of one line sits at the height of the start of the next a
    chain walks across the gap and welds them together. Cleaner than sorting, and
    still wrong — `IMG_1209` came out as 17 lines with 68, 67, 70 and 103
    characters where a line holds 34.

    A line is therefore followed along *itself*: from each character, look only
    ahead in the direction the line has been travelling, and re-estimate that
    direction from the last few characters read. The next line is never ahead in
    the direction the current one is going, so the jump cannot happen. It also
    tracks a line that bends, which the ruling on a curled sheet does.
    """
    pts = _centres(boxes)
    if len(pts) < MIN_LINES * MIN_CHARS_PER_LINE:
        return []
    med_h = float(np.median([p[2] for p in pts])) or 1.0
    xy = np.array([[p[0], p[1]] for p in pts], float)
    n = len(pts)
    order = np.argsort(xy[:, 0])                     # right to left is fine either way
    taken = np.zeros(n, bool)

    reach_x = 2.0 * med_h        # how far ahead to look for the next character
    corridor = 0.45 * med_h      # how far off the predicted height it may sit
    groups = []

    # Follow each line along itself, predicting where it goes next from the slope
    # of what has been read so far. Linking a character to anything nearby is not
    # enough: the links are transitive, and on a page tilted enough to matter the
    # end of one line sits at the height of the start of the next, so a chain
    # walks across the gap and welds two lines into one. Following a *direction*
    # cannot make that jump, because the next line is never ahead in the
    # direction the current one is travelling.
    for seed in order:
        if taken[seed]:
            continue
        line = [seed]
        taken[seed] = True
        for direction in (1, -1):
            slope = 0.0
            cx, cy = xy[seed]
            while True:
                ahead = (xy[:, 0] - cx) * direction
                dy = xy[:, 1] - (cy + slope * ahead * direction)
                ok = (~taken) & (ahead > 0) & (ahead <= reach_x) & (np.abs(dy) <= corridor)
                if not ok.any():
                    break
                cand = np.where(ok)[0]
                nxt = cand[np.argmin(ahead[cand])]
                line.append(nxt)
                taken[nxt] = True
                cx, cy = xy[nxt]
                # Re-estimate the direction from the tail of the line, so the fit
                # tracks a line that bends instead of drifting off it.
                tail = line[-8:] if direction == 1 else line[-8:]
                if len(tail) >= 3:
                    t = xy[tail]
                    if t[:, 0].max() - t[:, 0].min() > 1e-6:
                        slope = float(np.polyfit(t[:, 0], t[:, 1], 1)[0]) * direction
        if len(line) >= MIN_CHARS_PER_LINE:
            groups.append([pts[i] for i in sorted(line, key=lambda i: xy[i, 0])])

    groups.sort(key=lambda g: float(np.mean([p[1] for p in g])))

    # Drop the fragments.
    #
    # Crowns on the letters and the ruled lines themselves throw off short
    # chains that sit almost on top of a real line: on one engraved mezuza this
    # returned 29 lines for a page of 22, and the spurious ones showed up as
    # gaps of 0, 1, 5 and 7 px where a line is 120 px from its neighbour. That
    # matters more than it looks, because the horizon below is found by making
    # the line spacing as even as possible, and half of what it was evening out
    # was noise. It rectified a page that measured 1.07 degrees into one that
    # measured 2.18.
    if len(groups) >= MIN_LINES:
        sizes = [len(g) for g in groups]
        floor = 0.4 * float(np.median(sizes))
        groups = [g for g in groups if len(g) >= floor]
    return groups


def _fit(pts: np.ndarray) -> np.ndarray:
    """Least-squares line through points, as homogeneous (a, b, c)."""
    x, y = pts[:, 0], pts[:, 1]
    if x.max() - x.min() >= y.max() - y.min():
        m, c = np.polyfit(x, y, 1)
        line = np.array([m, -1.0, c])
    else:
        m, c = np.polyfit(y, x, 1)
        line = np.array([1.0, -m, -c])
    n = np.linalg.norm(line[:2])
    return line / (n or 1.0)


def _intersect(lines: list[np.ndarray]) -> np.ndarray:
    """Least-squares intersection of homogeneous lines."""
    _, _, vt = np.linalg.svd(np.array(lines))
    return vt[-1]


def fan_degrees(groups: list) -> float:
    """Angle between the topmost text line and the bottom one.

    Estimated by the median of the pairwise rates rather than by a least-squares
    fit. One short line whose angle is off — and a page of script always has a
    few, the last line of a paragraph among them — swings a least-squares slope
    hard when it sits at the top or the bottom of the block, which is exactly
    where it has the most leverage. `mezuzah1` measured 0.97 degrees with one
    grouping and 2.92 with another that differed by a single line, and the
    second was enough to send a flat page through the correction.
    """
    if len(groups) < MIN_LINES:
        return 0.0
    angs, ys = [], []
    for g in groups:
        pts = np.array([[p[0], p[1]] for p in g])
        if pts[:, 0].max() - pts[:, 0].min() < 1e-6:
            continue
        angs.append(np.degrees(np.arctan(np.polyfit(pts[:, 0], pts[:, 1], 1)[0])))
        ys.append(float(pts[:, 1].mean()))
    if len(angs) < MIN_LINES:
        return 0.0
    a, y = np.array(angs), np.array(ys)
    rates = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            dy = y[j] - y[i]
            if abs(dy) > 1e-6:
                rates.append((a[j] - a[i]) / dy)
    if not rates:
        return 0.0
    return abs(float(np.median(rates)) * (y.max() - y.min()))


def _horizon(baselines: list, v_text: np.ndarray) -> np.ndarray | None:
    """The vanishing line, from the fact that ruled lines are evenly spaced.

    The horizon has to pass through the text's vanishing point, which leaves one
    degree of freedom. Pinning it needs a second measurement, and the obvious
    one — where the left-hand margin converges with the right — is not available
    here: STaM justifies the right margin and lets the left fall where the words
    end. Measured across these pages the right margin sits within 12-32 px of a
    straight line and the left within 66-132, so half the fit was noise, and the
    correction it produced moved a corner by 49-65% of the image.

    Line *spacing* is the sounder constraint. The lines are ruled into the skin
    before a word is written, so on the parchment they are parallel and evenly
    spaced; in the photograph they crowd together as they recede, at a rate that
    fixes the horizon exactly. Twenty ruled lines, not two ragged margins.
    """
    vx, vy, vw = float(v_text[0]), float(v_text[1]), float(v_text[2])

    # Each near-horizontal baseline meets the vertical axis at a height that is
    # all the projective row of the matrix acts on, so the search is over one
    # scalar and the rest follows from the constraint.
    heights = []
    for line in baselines:
        if abs(line[1]) < 1e-9:
            continue
        heights.append(-line[2] / line[1])
    if len(heights) < MIN_LINES:
        return None
    heights = np.sort(np.array(heights))

    # A mezuza breaks into paragraphs, and the gap at a break is not the gap
    # between two lines of the same paragraph. Judging evenness on the median
    # rather than the mean keeps those breaks from dragging the fit: a handful
    # of wide gaps move a median hardly at all and a standard deviation a great
    # deal.
    def unevenness(b: float) -> float:
        denom = b * heights + 1.0
        if np.any(np.abs(denom) < 1e-6):
            return np.inf
        y = heights / denom
        d = np.diff(np.sort(y))
        if len(d) < 2 or np.any(d <= 0):
            return np.inf
        med = float(np.median(d))
        if med <= 0:
            return np.inf
        return float(np.median(np.abs(d - med)) / med)

    span = max(abs(heights).max(), 1.0)
    grid = np.linspace(-4.0 / span, 4.0 / span, 801)
    scores = [unevenness(b) for b in grid]
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        return None
    # refine around the winner
    lo = grid[max(0, best - 1)]
    hi = grid[min(len(grid) - 1, best + 1)]
    fine = np.linspace(lo, hi, 201)
    b = float(fine[int(np.argmin([unevenness(v) for v in fine]))])

    # The horizon passes through the text vanishing point: a*vx + b*vy + vw = 0.
    if abs(vx) < 1e-9:
        return None
    a = -(b * vy + vw) / vx
    return np.array([a, b, 1.0])


def homography(groups: list, shape: tuple) -> np.ndarray | None:
    """Map the keystoned page onto a flat one, or None if it cannot be trusted."""
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Centre first: the projective row of the matrix is tiny, and evaluating it
    # at coordinates in the thousands loses the precision it needs.
    T = np.array([[1.0, 0, -cx], [0, 1.0, -cy], [0, 0, 1.0]])

    baselines, right = [], []
    for g in groups:
        pts = np.array([[p[0] - cx, p[1] - cy] for p in g])
        baselines.append(_fit(pts))
        right.append(pts[np.argmax(pts[:, 0])])

    v_text = _intersect(baselines)
    horizon = _horizon(baselines, v_text)
    if horizon is None:
        return None
    Hp = np.array([[1.0, 0, 0], [0, 1.0, 0], [horizon[0], horizon[1], 1.0]])

    # The column direction comes from the justified margin alone, measured after
    # the projective part has been undone, where it is a straight line again.
    rp = np.array(right, float)
    moved = cv2.perspectiveTransform(rp.reshape(-1, 1, 2).astype(np.float32),
                                     Hp).reshape(-1, 2)
    if not np.all(np.isfinite(moved)):
        return None
    col = _fit(moved)
    v_col = np.array([-col[1], col[0], 0.0])          # direction along that margin

    d1, d2 = Hp @ v_text, v_col
    n1, n2 = np.linalg.norm(d1[:2]), np.linalg.norm(d2[:2])
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    d1, d2 = d1[:2] / n1, d2[:2] / n2
    # See the module docstring: without this the page is rectified into its
    # mirror image, which no metric we have would catch.
    if d1[0] < 0:
        d1 = -d1
    if d2[1] < 0:
        d2 = -d2

    B = np.array([[d1[0], d2[0]], [d1[1], d2[1]]])
    if abs(np.linalg.det(B)) < 1e-6:
        return None
    A = np.linalg.inv(B)
    det = np.linalg.det(A)
    if det <= 0:
        return None                       # left-handed: would mirror the text
    A = A / np.sqrt(det)                  # unit determinant keeps letters the same size

    Ha = np.eye(3)
    Ha[:2, :2] = A
    return np.linalg.inv(T) @ Ha @ Hp @ T


def apply_to_boxes(boxes: list, H: np.ndarray) -> list:
    """Carry the character boxes through the same warp."""
    out = []
    for b in boxes:
        v = b.get("vertices") or []
        if not v:
            continue
        pts = np.array([[[p.get("x", 0), p.get("y", 0)]] for p in v], np.float32)
        moved = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        out.append({**b, "vertices": [{"x": float(px), "y": float(py)}
                                      for px, py in moved]})
    return out


def rectify(img: np.ndarray, boxes: list, fill=None):
    """Return (image, H, info). H is None when the page was left untouched."""
    info: dict = {"fan": 0.0, "rectified": False, "reason": ""}
    if not ENABLED or img is None or not boxes:
        info["reason"] = "off" if not ENABLED else "no boxes"
        return img, None, info

    groups = group_lines(boxes)
    info["lines"] = len(groups)
    if len(groups) < MIN_LINES:
        info["reason"] = "too few text lines"
        return img, None, info

    info["fan"] = round(fan_degrees(groups), 2)
    if info["fan"] < MIN_FAN:
        info["reason"] = "flat enough"
        return img, None, info

    H = homography(groups, img.shape)
    if H is None:
        info["reason"] = "fit rejected"
        return img, None, info

    # Check the answer against the question. Carrying the character boxes
    # through the transform and measuring them again costs a millisecond and
    # says plainly whether the page came out straighter than it went in. A fit
    # that does not flatten the page is a fit that has gone wrong, whatever its
    # corners happen to do.
    after = fan_degrees(group_lines(apply_to_boxes(boxes, H)))
    info["fan_after"] = round(after, 2)
    if after > max(0.4 * info["fan"], 0.5):
        info["reason"] = f"still {after:.1f}° after correction, refusing"
        return img, None, info

    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    shift = np.abs(moved - corners.reshape(-1, 2)).max() / max(w, h)
    if shift > MAX_SHIFT:
        info["reason"] = f"corner moved {shift:.0%}, refusing"
        return img, None, info

    # Frame the result on the writing, not on the warped rectangle.
    #
    # A projective warp turns the crop rectangle into a quadrilateral, and
    # taking its bounding box leaves wide wedges of blank parchment in the
    # corners: on IMG_1213 the text fell from 94% of the image to 58%, with a
    # 356 px margin down one side. The engine noticed before we did — it logged
    # `init_new_word was called when last word in line is empty` eleven times on
    # that image, looking for words in the empty ground, and its error count rose
    # from 467 to 547 on a page that had just been straightened.
    ink = cv2.perspectiveTransform(
        np.array([[[p[0], p[1]]] for g in groups for p in g], np.float32),
        H).reshape(-1, 2)
    pad = 0.5 * float(np.median([p[2] for g in groups for p in g]))
    x0 = max(float(moved[:, 0].min()), float(ink[:, 0].min()) - pad)
    y0 = max(float(moved[:, 1].min()), float(ink[:, 1].min()) - pad)
    x1 = min(float(moved[:, 0].max()), float(ink[:, 0].max()) + pad)
    y1 = min(float(moved[:, 1].max()), float(ink[:, 1].max()) + pad)
    ow, oh = int(round(x1 - x0)), int(round(y1 - y0))
    if not (0 < ow < 4 * w and 0 < oh < 4 * h):
        info["reason"] = f"implausible output {ow}x{oh}"
        return img, None, info

    S = np.array([[1.0, 0, -x0], [0, 1.0, -y0], [0, 0, 1.0]])
    total = S @ H
    if fill is None:
        fill = tuple(int(v) for v in np.median(img.reshape(-1, img.shape[2]), 0))
    out = cv2.warpPerspective(img, total, (ow, oh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    info["rectified"] = True
    info["reason"] = "ok"
    info["out"] = f"{ow}x{oh}"
    return out, total, info
