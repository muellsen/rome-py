# Attribution and reference materials

The exact Huber coordinate-search principle is due to Younghoon Kim,
Po-Ling Loh and Sumanta Basu (arXiv:2510.13715). The package implementation
is independently written; the original R/C package is downloaded and built
separately for comparison. Its DESCRIPTION specifies GPL-3.

The estimator follows Patrick L. Combettes and Christian L. Müller,
“Perspective M-estimation via proximal decomposition” (2020).
The perspective construction and mean-shift representation are existing work.

The audited c-lasso 1.0.11 files under sources/classo-1.0.11-audit are third-party
reference copies, distributed under their accompanying MIT license.
They are excluded from the Python wheel and retain their upstream attribution.
Our package does not import them at runtime.

Downloaded papers, datasets, publisher templates and archives retain their
original rights and are not included in the Git submission or Python wheel.
Source URLs and checksums are retained so they can be retrieved separately.
The Taylor & Francis template is a reference download, not a verified claim
that its present layout satisfies JCGS submission requirements.

The local AISTATS AMP reference and user-provided drafts are excluded from the
Git submission. Optional AMP experiments require those local reference files.
Their exclusion does not alter the original recorded local provenance.

The PDF algorithm environment uses algorithm2e, ifoddpage, relsize and the
ydoc extraction utility from CTAN under their respective LaTeX Project Public
Licenses. These downloaded files remain in an ignored local vendor cache.
sources/algorithm-template.json pins their archives; fetch_algorithm_template.py
reconstructs the cache. No publisher artwork or manuscript prose is redistributed.
