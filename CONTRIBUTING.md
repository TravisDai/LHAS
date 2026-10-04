# Contributing to LHAS

Thank you for helping improve the reproducibility of LHAS. Questions, defect
reports, and documentation corrections can be submitted through
[GitHub Issues](https://github.com/TravisDai/LHAS/issues).

## Reproduction questions and defect reports

Include the repository commit (`git rev-parse HEAD`), operating system, Python
version, installation command, complete command that failed, and the relevant
log excerpt. State what you expected and what happened. For result discrepancies,
identify the reference JSON file, workload, node count, and whether the run uses
analytical or measured computation inputs. Remove credentials and unrelated
personal information from logs before sharing them.

Start with the [small reproduction check](README.md#quick-start-cpu-only) when
possible. The [reproduction guide](docs/REPRODUCING.md) explains optional
dependencies and common sources of differences.

## Proposed changes

Use a focused branch and pull request. Explain the issue, the resulting behavior,
and the checks you ran. Keep documentation changes separate from model changes
where practical, and use American English in reader-facing documentation.

The recorded experiments are research evidence:

- Preserve stored raw results, measured ZIPs, workload manifests, and provenance.
  Write new experiments to a separate output directory or tag.
- Identify any change to model assumptions, candidate domains, statuses, or
  numerical results explicitly. Do not tune a correction toward an earlier result.
- For a code defect, add a focused regression that demonstrates the defect and
  run the relevant existing checks. Documentation-only changes normally need link
  and command checks rather than new unit tests.
- Report skipped or unavailable checks. CPU-only tests do not establish GPU
  profiling correctness or optical hardware accuracy.
- Keep unpublished manuscript sources, bibliography files, manuscript-only
  figures, and review correspondence out of public commits and attachments.

The software license is still pending. This guide does not establish a new
licensing agreement; confirm the contribution and reuse terms with the maintainer
before contributing third-party material.
