# Line-Sweep Alignment with Shift-on-Crossing Rule (Legacy Design)

## Problem Overview

We are given **n text boxes**. Each box `i` has:

- A **left edge** `L_i` (x-coordinate, pixels)
- A list of **candidate x-positions** for alignment, strictly increasing:
  ```
  X_i = [x_{i,0}, x_{i,1}, ..., x_{i,m_i}]
  ```
  where:
  - `x_{i,0}` is the current/default edge
  - `x_{i,1}` is the *next candidate to the right*, etc.

The goal is to place a **vertical alignment line** at position `line_x` and select one candidate per box such that:
- The selected edges are as **conforming** as possible
- Moving the line to the right may **force boxes to shift**
- Each forced shift has a cost

---

## Key Asymmetry: Shift Is Defined by Geometry

There is **no separate “drop” or “shift” decision**.

A box is considered **shifted** if and only if the alignment line crosses its left edge.

Formally:

- **Unshifted / included**
  ```
  line_x ≤ L_i  ⇒  y_i = x_{i,0}
  ```

- **Shifted / altered**
  ```
  line_x > L_i  ⇒  y_i = x_{i,k_i},  k_i ≥ 1
  ```

So:
- Crossing a left edge **forces** a shift
- The number of shifted boxes is:
  ```
  A(line_x) = Σ_i 1[ line_x > L_i ]
  ```

This asymmetry is central to the algorithm.

---

## Objective Function

Let `y = (y_1, ..., y_n)` be the chosen edges implied by `line_x`.

### Conformity term
Use quadratic distance to the current line:

```
Q(line_x) = (1/n) * Σ_i (y_i - line_x)^2
```

### Shift cost
```
A(line_x) = Σ_i 1[ line_x > L_i ]
```

### Total objective
```
J(line_x) = Q(line_x) + λ * A(line_x)
```

Optionally, use a percentage form:
```
J(line_x) = Q(line_x) + λ * (A(line_x) / n)
```

---

## Core Idea: Move the Line, Not the Boxes

Instead of testing subsets or box states, we **move the alignment line `line_x` from left to right** and observe how the objective changes.

### Key observation

- The shift status of a box changes **only when `line_x` crosses its left edge `L_i`**
- Between two consecutive left edges, nothing changes
- Therefore, the objective changes at only **n discrete events**

This enables an **event-driven line sweep**.

---

## Line-Sweep Algorithm

### Step 1: Sort events
Sort boxes by their left edges:
```
L_(1) ≤ L_(2) ≤ ... ≤ L_(n)
```

Each `L_(k)` is an event where one more box becomes shifted.

---

### Step 2: Initialize (line far left)

Start with `line_x` left of all boxes:

- No box is shifted
- For all `i`: `y_i = x_{i,0}`
- Shift count `A = 0`
- Compute initial `Q(line_x)` and `J`

---

### Step 3: Sweep the line right

For `k = 1 .. n`:

- Move `line_x` just past `L_(k)`
- Box `(k)` becomes shifted
- Increment:
  ```
  A ← A + 1
  ```
- Update its chosen value:
  ```
  y_(k): x_(k,0) → x_(k,1) (or another allowed candidate ≥ 1)
  ```
- Update:
  - `Q(line_x)`
  - `J = Q(line_x) + λ * A`

Only **one box changes** at each event.

---

### Step 4: Candidate selection when shifting

When a box becomes shifted, choose `y_i` from `{x_{i,1} ... x_{i,m_i}}`.

Two standard strategies:

#### Variant A: Next-only (fastest)
Always take:
```
y_i = x_{i,1}
```

- O(1) per event
- Deterministic and simple

#### Variant B: Best-fit (more accurate)
Choose the candidate closest to the current target (e.g. current line):

```
y_i = argmin |x - line_x|
```

- Candidates are sorted → binary search
- O(log m_i) per event

---

## Efficient Quadratic Updates

Maintain running sums:

```
S1 = Σ y_i
S2 = Σ y_i^2
```

Then:
```
Q(line_x) = (S2 / n) - 2 * line_x * (S1 / n) + line_x^2
```

When a single value changes from `a` to `b`:
```
S1 ← S1 + (b - a)
S2 ← S2 + (b^2 - a^2)
```

So each event update is **O(1)** once `b` is chosen.

---

## Stopping Strategy

Two options:

### Greedy stop
- Stop at the first event where `J` does not improve

### Recommended: sweep all, pick best
- Sweep all events
- Track the minimum `J`
- Return the best `line_x` / configuration

This avoids local premature stopping.

---

## Complexity

Let:
- `n` = number of boxes
- `m` = average number of candidates per box

Costs:
- Sorting left edges: `O(n log n)`
- Sweep:
  - Next-only: `O(n)`
  - Best-fit: `O(n log m)`

Total:
- `O(n log n)` (next-only)
- `O(n log n + n log m)` (best-fit)

Memory: `O(n)`

---

## Edge Cases & Guardrails

- If a box has no shifted candidate (`m_i = 0`):
  - Once `line_x > L_i`, the configuration is invalid
  - Either stop the sweep or forbid crossing that edge
- If multiple boxes share the same `L_i`, treat as a batch event
- You do not need a continuous `line_x`; the state is defined by how many edges were crossed

---

## Interpretation

- The algorithm reduces a combinatorial selection problem to a **1D sweep**
- Asymmetry (left-edge crossing) is what makes this possible
- Each box transitions state exactly once
- The algorithm is fast, deterministic, and easy to debug

Generalization:
- For **right** or **bottom** alignment, mirror coordinates (x'=-x or y'=-y) and apply the same sweep.
- Candidates should be monotone toward the interior for the chosen side.

This document captures the intended legacy design and rationale.
