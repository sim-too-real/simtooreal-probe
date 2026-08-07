# Publishing this package as a public GitHub repo

This tree is intentionally **outside** the private platform monorepo so it can
be published alone under Apache-2.0.

## One-time publish checklist

1. Confirm open-source readiness: `pytest`, `python -m build`, no secrets.
2. Create an empty public repo: **`sim-too-real/simtooreal-probe`**.
3. From this directory:

```bash
git init
git add .
git commit -m "Initial open-core simtooreal-probe (policy debugging layer)"
git branch -M main
git remote add origin git@github.com:sim-too-real/simtooreal-probe.git
git push -u origin main
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

4. Enable GitHub Actions (`.github/workflows/ci.yml`).
5. Publish to PyPI when ready:

```bash
python -m build
python -m twine upload dist/*
```

Until PyPI is live, consumers install from git:

```bash
pip install "git+https://github.com/sim-too-real/simtooreal-probe.git@v0.1.0"
```

## Multi-root workspace (platform + OSS)

If you also develop the private platform monorepo, open a multi-root workspace
that includes:

- the private monorepo (platform)
- this package (`simtooreal-probe`)
- optional: `simtooreal-cli`

A convenience file may live next to both trees as
`simtooreal-workspace.code-workspace`.

## Local editable install

```bash
pip install -e path/to/simtooreal-probe
# Platform training-host package (private) depends on this package:
# pip install -e path/to/monorepo/simtooreal-tooling
```

Training images can vendor a git pin or a built wheel until PyPI is available.
