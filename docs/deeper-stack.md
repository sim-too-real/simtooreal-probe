# The deeper stack under RoboSynx and SimTooReal

Public robotics AI products often stop at demos: a clean robot file, a training curve, a video of a policy in sim. The harder questions arrive in diligence:

- What is actually *admitted* as physical knowledge?
- Where do mechanical claims live, with units and provenance?
- Who is allowed to command anything near metal — and can a learned system upgrade a refuse?

This post is the sincere map of our deeper stack: **First Principles**, **engineering-data**, and **reality-os-rust** — and how they relate to the public surfaces **RoboSynx** and **SimTooReal**.

Contact: **vardhan@simtooreal.com** · [simtooreal.com](https://simtooreal.com) · [robosynx.com](https://robosynx.com)

## The public surface first

| Layer | What it is | Link |
|-------|------------|------|
| Robot files & free tools | Generate / convert / validate / view URDF·SDF·MJCF; ROS 2 scaffold | [robosynx.com](https://robosynx.com) |
| Training ops & transfer | Isaac Lab / MuJoCo metrics, failure detection, five-pillar transfer scoring | [simtooreal.com](https://simtooreal.com) |

Those are what you can try without a sales call. The rest of this article is what serious buyers ask about next. First Principles and engineering-data are **private R&D / pilot diligence** unless we publish public docs. reality-os-rust is a **private GitHub** repo (`sim-too-real/reality-os-rust`, MIT) for the same diligence lane — not a “clone and ship metal” promise.

## Layer diagram (honest)

```
  RoboSynx          SimTooReal
  (files / free)    (train / score)
         \              /
          \            /
           v          v
     Physical AI product path
              |
    +---------+---------+
    |                   |
    v                   v
 First Principles     engineering-data
 (mechanism           (claims, rights,
  admission,           provenance, SI,
  machine::verify)     curves; never
                       writes AdmissionStatus)
    |                   |
    +--------+----------+
             |
             v
      reality-os-rust
      (fail-closed Governor:
       allow / narrow / abort only;
       SIM ≠ metal)
```

Public writing about all three deeper components is **encouraged** when it stays diligence-honest: no fake ISO/SIL, no metal MEASURED theater, no invent-any-machine claims.

---

## 1) First Principles — admitted knowing as mechanisms

**Job:** Growing source of truth for admitted definitions, conservation laws, unit-checked equations, and passed/failed experiments.

**Unit of knowing:** a **mechanism** — inputs, outputs, invariants, test — living under `mechanisms/`. English is interface only; it is never how truth is stored.

**Rust `kernel/`:** SI `Qty` / `Dim`, SI 2019 constants, honesty stamp, mechanism parser. Bare numbers without units are refused.

**Domains:** `math`, `phys`, `chem`, `bio`, `fp`, `machine` — a reduction stack. Layers cannot be skipped.

**Simulation admission:** `machine::verify` is the sole simulation-admission authority. A FreeCAD add-on may call the FP workbench; the **viewport never admits**. No executable G-code. Path/CAM is **named-missing**.

**Official vertical:** unpowered pinned structural linkage. **Program step:** one real physical lifecycle.

First Principles answers: *may this be admitted as knowing / simulation?*

---

## 2) engineering-data — claims with provenance, not a second kernel

**Job:** Machine-usable claims about mechanical and industrial engineering — rights, provenance, SI quantities, curves.

**Hard rule:** It is **not** a second First Principles kernel. It **never** writes `AdmissionStatus`.

**v1:** local CAS + SQLite. CLI: `ed ingest` / `query` / `gaps` / `report` / `provenance` / `pack`.

**Fixtures:** material fixtures in v1 are **synthetic** — not AISI allowables. Cross-repo verify with First Principles keeps the boundary clear.

engineering-data answers: *what claims do we hold, with what trail and units?*

---

## 3) reality-os-rust — last gate before anyone confuses SIM with metal

**Job:** Rust last-gate for Reality OS + Governor (`theworld-runtime` port of the authority kernel).

**Description string:** *Rust last-gate for Reality OS + Governor: fail-closed certified command path. SIM ≠ metal.*

**Governor:** may only allow, narrow, or abort. Cannot invent a plan or upgrade a refuse. Learned systems never write motors.

**Crates:** `realityos-kernel`, `physics`, `data`, `plant`, `governor`, `core`, `session`, `ros2`, `vport`, `hil`, plus `ros-governor` CLI.

**Honest ledger:**

| Topic | Public answer |
|-------|----------------|
| SIM last-gate | Yes |
| ONLINE metal / MEASURED PFL | No |
| ISO 13850 / 10218 / 26262 / SIL | No |
| Independent HW e-stop | No |
| Invent any machine | No |
| VLA last-write | No |

“Certified command path” here means fail-closed software authority design — **not** a completed third-party safety certificate.

reality-os-rust answers: *given a proposal, what is the governor allowed to do?*

---

## How a diligence conversation usually flows

1. Try **RoboSynx** — can we get a sane robot file into Isaac / MuJoCo / ROS 2 without install tax?
2. Try **SimTooReal** — can we see training failure modes and a transfer score before hardware?
3. Ask about **First Principles** — how are physical claims unit-checked and admitted?
4. Ask about **engineering-data** — where do mechanical claims and provenance live?
5. Ask about **Reality OS** — what happens when a learned policy wants motor authority?

Skipping from step 1 to metal is how teams create demos that cannot survive a safety or procurement review.

## What we will not say

- That First Principles or engineering-data are public GitHub products today (they are private R&D / pilot diligence unless we publish docs)
- That we hold ISO 13850 / 10218 / 26262 / SIL certificates we have not earned
- That SIM evidence is metal MEASURED
- That we invent arbitrary machines end-to-end
- That a viewport or a neural net is an admission authority

## Pilots and contact

Fixed-scope SimTooReal pilots remain the commercial on-ramp (connect one GPU host, instrument one training project, written readout). Reality OS and the deeper admission stack are discussed as diligence / upsell for hardware-bound teams — with the honesty ledger intact.

**vardhan@simtooreal.com**

- Training & scoring: [simtooreal.com](https://simtooreal.com)
- Free robot-file tools: [robosynx.com](https://robosynx.com)

Calm gates beat loud demos. We intend to keep building that way.
