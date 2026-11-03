# GitHub repository

Repository and distribution: `rome-py`; Python import: `rome_py`.
Display name: ROME for Python.

The maintainer selected this name on 2026-11-03. PyPI's JSON API returned
404 for `rome-py` at that time; this does not reserve the name or guarantee
acceptance. Both `pyrome` and `romepy` are occupied by unrelated projects.

The public interface in `src/rome_py` re-exports the numerical implementation
in `src/concomitant_huber`. Existing imports, solver files, and historical
experiment manifests remain unchanged.

## Prepared contents

- Installable src-layout package, GPL-3.0-only license, attribution, and tests.
- GitHub Actions for package checks, distributions, and a public research workflow.
- Quarto manuscript, experiment drivers, source registries, and provenance tools.
- Ignore rules and a submission audit excluding private inputs, downloaded
  references, numerical caches, and generated manuscripts.

Run `python analysis/export_submission.py` to audit and create the source-only
GitHub ZIP. Package builds contain the numerical library and public interface;
research sources are distributed through the repository.

The repository starts private. Before public release, finalize citation authors
and review the manuscript and research artifacts. PyPI publication is separate.

Naming reference: https://packaging.python.org/en/latest/specifications/name-normalization/
