# Mech OS Datastack — machine-usable claims, not a second Mech OS kernel

Industrial and mechanical engineering already have enormous knowledge. Most of it is trapped in PDFs, tribal spreadsheets, and tribal memory. Physical AI teams need something else: **machine-usable claims** — rights, provenance, SI quantities, and curves — that a program can query without pretending English is a database.

**Mech OS Datastack** is that claims store. It is private R&D / pilot diligence today. It is deliberately **not** a second Mech OS kernel.

Contact: **vardhan@simtooreal.com** · [simtooreal.com](https://simtooreal.com) · [robosynx.com](https://robosynx.com)

## What it stores

Mech OS Datastack holds machine-usable claims about mechanical and industrial engineering:

- **rights** — what you are allowed to use and under what conditions the claim was recorded
- **provenance** — where a claim came from, so audits are possible
- **SI quantities** — numbers with units, not orphan floats
- **curves** — engineering curves as first-class data, not screenshots

The goal is a queryable substrate for diligence and tooling — not a blog of engineering tips.

## Explicit non-goals

Two lines we will not blur:

1. **Not a second Mech OS kernel.** Mech OS owns mechanism admission, conservation structure, and the honesty stamp for admitted knowing. Mech OS Datastack does **not** write `AdmissionStatus`.
2. **Fixture materials in v1 are synthetic.** They are not AISI allowables. If a demo curve looks like a steel grade table, treat it as a fixture unless provenance says otherwise.

Verification with Mech OS exists so the claims store and the admission kernel stay honest about their division of labor.

## v1 shape: local CAS + SQLite

The v1 architecture is intentionally boring and inspectable:

- **Local content-addressed storage (CAS)** for blobs and artifacts
- **SQLite** for queryable claim indexes and metadata
- A **CLI** surface:

| Command | Role |
|---------|------|
| `ed ingest` | Bring claims / artifacts in |
| `ed query` | Ask the store |
| `ed gaps` | Surface what is missing |
| `ed report` | Human-readable summaries |
| `ed provenance` | Trace origin |
| `ed pack` | Package for diligence / transfer |

Boring storage is a feature. Auditors and pilot partners should be able to reason about the disk layout without a mystery service.

## Why separate it from Mech OS

Mech OS answers: *is this a unit-checked mechanism that may be admitted?*

Mech OS Datastack answers: *what mechanical/industrial claims do we have, with what rights and provenance, expressed in SI?*

Mixing those jobs into one process creates a temptation: let a data ingest silently become an admission. We refuse that. Mech OS Datastack never writes `AdmissionStatus`. Mech OS remains the admission authority for knowing; `machine::verify` remains the simulation-admission authority on the Mech OS side.

## How teams should use this in diligence

If you are evaluating SimTooReal / RoboSynx and ask “what about mechanical allowables and claim provenance?”, the honest answer is:

- Mech OS Datastack is the **claims and provenance layer**
- Mech OS is the **mechanism admission layer**
- Neither replaces a certified materials database you already trust for production metal decisions
- Synthetic fixtures in v1 are labeled as such — do not treat them as AISI allowables

That sentence is longer than a slogan. It is also true.

## Relation to the public products

[RoboSynx](https://robosynx.com) helps you get robot files into simulators without install tax. [SimTooReal](https://simtooreal.com) helps you monitor Isaac Lab / MuJoCo training and score transfer readiness.

Mech OS Datastack is deeper stack: the industrial-claim substrate behind diligence conversations. Public writing about it is encouraged when it stays diligence-honest — provenance, SI, synthetic fixtures named as synthetic, no fake certifications.

## Contact

**vardhan@simtooreal.com**

- [simtooreal.com](https://simtooreal.com)
- [robosynx.com](https://robosynx.com)

Claims without provenance are rumors. We prefer queries with a trail.
