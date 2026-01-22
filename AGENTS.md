# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains only image assets intended for benchmarking cropper behavior.

- `benchmark/`: Sample images used for manual or automated evaluation.
- There is no application source code, test suite, or build output tracked here.

If you add code later, keep it in a top-level `src/` folder and place tests in `tests/` so the structure stays predictable.

## Build, Test, and Development Commands

There are no build or test commands defined in this repository yet. If you introduce tooling, document it here with short examples, such as:

```bash
npm test        # run unit tests
python -m pytest  # run Python tests
```

## Coding Style & Naming Conventions

No coding conventions are defined because there is no code. For consistency with the existing assets:

- Keep image files in `benchmark/`.
- Prefer descriptive, lowercase ASCII filenames if you add new assets (e.g., `benchmark/scan-001.jpg`).
- If non-ASCII names are required, keep them consistent with the existing naming scheme.

## Testing Guidelines

No tests exist. If you add tests in the future, document:

- The testing framework used.
- The naming pattern (e.g., `test_*.py`, `*.spec.ts`).
- How to run the suite locally.

## Commit & Pull Request Guidelines

There is no commit history to infer conventions. Until a standard is agreed:

- Use clear, imperative commit messages (e.g., `Add benchmark images for mezuzah scans`).
- For pull requests, include a short description of the change and list any new assets or tooling.

## Assets & Usage Notes

- Images in `benchmark/` may include scans or photos; treat them as potentially sensitive.
- Do not overwrite or rename existing assets without noting the reason in your change description.
