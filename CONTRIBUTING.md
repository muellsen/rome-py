# Development

Install the package with benchmark dependencies, then run make check.
Keep numerical solver updates in src/concomitant_huber and experiment drivers
in analysis. The canonical manuscript is paper/manuscript.qmd.
Use the repository skills named in AGENTS.md for solver and R4 work.

Changes to numerical results require an experiment script, saved input arrays,
explicit settings and a provenance manifest. Preserve failed runs and distinguish
a declared validation subset from a completed replication study. A prose-only
edit must not silently replace the hashes attached to historical numerical runs.

The two main experiments have equal standing: original fixed-scale ROME versus
the concomitant extension, and unconstrained R4 solver scaling. Start the R4
scaling study with Path-Alg. Do not add other packages to the ROME track.
Keep fixed-scale behavior available.

Before opening a pull request, run tests relevant to the change, build a wheel,
and check the installed-package example. Do not commit downloaded references,
local drafts, private AMP source, numerical caches or generated manuscripts.
Use a reviewed release artifact for generated results and their manifests.
