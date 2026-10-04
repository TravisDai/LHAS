# Citing LHAS

Please identify the software checkpoint used in your work. The accompanying
TOMPECS manuscript is not yet published, and this repository does not currently
have an archival DOI or a tagged software release.

For now, include:

- **Software:** LHAS research reproducibility artifact.
- **Repository:** https://github.com/TravisDai/LHAS.
- **Version:** the full commit hash of your checkout (`git rev-parse HEAD`).
- **Access date:** the date you obtained that version.

Link to the corresponding immutable tree using
`https://github.com/TravisDai/LHAS/tree/<full-commit-hash>` rather than relying
only on the moving default branch.

When describing a result, also identify its raw JSON path, workload, system size,
and input mode. Stored experiment outputs retain their original development
commit hashes; [PROVENANCE.md](PROVENANCE.md) explains their relation to the public
checkpoint. Do not replace those recorded hashes with the current documentation
commit.

An author-approved paper citation and structured `CITATION.cff` can be added when
the bibliographic metadata are available. Do not infer a publication year,
article number, DOI, or acceptance status from this repository. Citation guidance
does not replace the pending software-license decision.
