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
Vision has returned a quad for every character, and in a keystoned photograph:

* the text lines, parallel on the parchment, converge to a vanishing point;
* the margins do too, and STaM is justified to a degree ordinary handwriting is
  not, so the first and last character of each line give a second one.

Two vanishing points fix the distortion completely. The homography that sends
them both to infinity makes parallel lines parallel again; a second, affine step
makes the two directions perpendicular. Both are a few matrix operations on
thirty points — the cost is one `warpPerspective`, 9 ms on a 3 MP image, against
the 575 ms the Vision call already costs.

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
# the page is left alone. Set from the benchmark: 93% of images measure under
# 2°, and those are pages that are already flat — correcting them can only
# resample them for nothing. The two images that need this measure 8.5° and 10°.
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
    """Sort character centres into text lines by vertical position."""
    pts = _centres(boxes)
    if len(pts) < MIN_LINES * MIN_CHARS_PER_LINE:
        return []
    med_h = float(np.median([p[2] for p in pts])) or 1.0
    pts.sort(key=lambda p: p[1])
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[1] - cur[-1][1] <= 0.6 * med_h:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    return [g for g in groups if len(g) >= MIN_CHARS_PER_LINE]


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
    """Angle between the first text line and the last, slope removed."""
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
    return abs(float(np.polyfit(y, a, 1)[0] * (y.max() - y.min())))


def homography(groups: list, shape: tuple) -> np.ndarray | None:
    """Map the keystoned page onto a flat one, or None if it cannot be trusted."""
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Centre first: the projective row of the matrix is tiny, and evaluating it
    # at coordinates in the thousands loses the precision it needs.
    T = np.array([[1.0, 0, -cx], [0, 1.0, -cy], [0, 0, 1.0]])

    baselines, right, left = [], [], []
    for g in groups:
        pts = np.array([[p[0] - cx, p[1] - cy] for p in g])
        baselines.append(_fit(pts))
        right.append(pts[np.argmax(pts[:, 0])])
        left.append(pts[np.argmin(pts[:, 0])])

    v_text = _intersect(baselines)
    v_col = _intersect([_fit(np.array(right)), _fit(np.array(left))])

    horizon = np.cross(v_text, v_col)
    if abs(horizon[2]) < 1e-12:
        return None                       # already parallel; nothing to undo
    horizon = horizon / horizon[2]
    Hp = np.array([[1.0, 0, 0], [0, 1.0, 0], [horizon[0], horizon[1], 1.0]])

    d1, d2 = Hp @ v_text, Hp @ v_col
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

    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    shift = np.abs(moved - corners.reshape(-1, 2)).max() / max(w, h)
    if shift > MAX_SHIFT:
        info["reason"] = f"corner moved {shift:.0%}, refusing"
        return img, None, info

    x0, y0 = moved.min(0)
    x1, y1 = moved.max(0)
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
