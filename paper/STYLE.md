# Manuscript and writing template

Primary model: Grace Yoon, Christian L. Müller and Irina Gaynanova,
Fast computation of latent correlations, JCGS 30(4), 1249–1256 (2021),
doi:10.1080/10618600.2021.1882468; arXiv:2006.13875v2.
The author-supplied public LaTeX source was retrieved and inspected on 2026-10-30.
Its source archive and SHA-256 record are preserved in sources/private/style-templates/.
The source uses 12-point article, letter paper, approximately one-inch margins,
an abstract and keywords, and one-and-a-half spacing in the body. This is the
author manuscript template, not a claim to reproduce the publisher's typeset pages.

Writing model: short problem-first introduction; establish the existing
computational task; introduce the new construction through explicit building
blocks; present algorithms before their performance assessment; compare
numerical accuracy before runtime; conclude with practical scope and limits.
Avoid an implementation diary in the main text. Use connected mathematical
prose and restrained claims. Detailed audits belong in the reproducibility materials.

The algorithm model is Algorithm 1 of Kim, Loh and Basu, arXiv:2510.13715v3,
whose source uses algorithm2e with ruled/vlined options. Our PDF uses the same
algorithm environment and control-flow typography. Additions/replacements
for concomitant estimation are blue. No scale-stale sequential strong rule
is asserted. The exact coordinate subroutine is a separate mathematical algorithm.

Use sign/min for Huber scores, sigma^(0) for initial scale, rho for the
dimensionless Huber threshold, and lambda0 for the joint lasso penalty.
Do not add unverified claims from supplied drafts.
