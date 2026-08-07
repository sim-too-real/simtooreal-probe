# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Report privately via one of:

- GitHub Security Advisories on [sim-too-real/simtooreal-probe](https://github.com/sim-too-real/simtooreal-probe/security/advisories/new) (once the repo is public)
- Email the maintainers listed on the GitHub organization profile

Include:

- Affected package version / commit
- Reproduction steps or PoC
- Impact assessment (e.g. path traversal under `PROBE_HOME`, credential leakage via env)

We aim to acknowledge reports within 7 days and ship fixes as coordinated disclosures.

## Non-goals / trust model

- **Local store:** traces under `PROBE_HOME` / `~/.probe` are as sensitive as your training logs. Protect that directory on multi-user machines.
- **HTTP sink:** tokens are read from environment (`PROBE_HTTP_TOKEN`, `SIMTOOREAL_*`). Never commit tokens. Failures of remote ingest must not crash training.
- **HTML export:** `to_html` embeds episode data in a local file; treat exported HTML as sensitive if observations contain private scene content.
