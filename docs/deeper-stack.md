# The deeper stack under RoboSynx and SimTooReal

Public robotics AI products often stop at demos: a clean robot file, a training curve, a video of a policy in sim. The harder questions arrive in diligence:

- What is actually *admitted* as physical knowledge?
- Where do mechanical claims live, with units and provenance?
- Who is allowed to command anything near metal — and can a learned system upgrade a refuse?

This post is the sincere map of our deeper stack: **Mech OS**, **Mech OS Datastack**, and **Reality OS** — and how they relate to the public surfaces **RoboSynx** and **SimTooReal**.

Contact: **vardhan@simtooreal.com** · [simtooreal.com](https://simtooreal.com) · [robosynx.com](https://robosynx.com)

## The public surface first

| Layer | What it is | Link |
|-------|------------|------|
| Robot files & free tools | Generate / convert / validate / view URDF·SDF·MJCF; ROS 2 scaffold | [robosynx.com](https://robosynx.com) |
| Training ops & transfer | Isaac Lab / MuJoCo metrics, failure detection, five-pillar transfer scoring | [simtooreal.com](https://simtooreal.com) |

Those are what you can try without a sales call. The rest of this article is what serious buyers ask about next. Mech OS, Mech OS Datastack, and Reality OS are **private R&D / pilot diligence** unless we publish public docs — not a “clone and ship metal” promise.

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
   Mech OS         Mech OS Datastack
 (mechanism         (claims, rights,
  admission,         provenance, SI,
  machine::verify)   curves; never
                     writes AdmissionStatus)
    |                   |
    +--------+----------+
             |
             v
       Reality OS
      (fail-closed Governor:
       allow / narrow / abort only;
       SIM ≠ metal)
```

Public writing about all three deeper components is **encouraged** when it stays diligence-honest: no fake ISO/SIL, no metal MEASURED theater, no invent-any-machine claims.

---

## 1) Mech OS — admitted knowing as mechanisms

**Job:** Growing source of truth for admitted definitions, conservation laws, unit-checked equations, and passed/failed experiments.

**Unit of knowing:** a **mechanism** — inputs, outputs, invariants, test — living under `mechanisms/`. English is interface only; it is never how truth is stored.

**Rust `kernel/`:** SI `Qty` / `Dim`, SI 2019 constants, honesty stamp, mechanism parser. Bare numbers without units are refused.

**Domains:** `math`, `phys`, `chem`, `bio`, `fp`, `machine` — a reduction stack. Layers cannot be skipped.

**Simulation admission:** `machine::verify` is the sole simulation-admission authority. A FreeCAD add-on may call the Mech OS workbench; the **viewport never admits**. No executable G-code. Path/CAM is **named-missing**.

**Official vertical:** unpowered pinned structural linkage. **Program step:** one real physical lifecycle.

Mech OS answers: *may this be admitted as knowing / simulation?*

---

## 2) Mech OS Datastack — claims with provenance, not a second kernel

**Job:** Machine-usable claims about mechanical and industrial engineering — rights, provenance, SI quantities, curves.

**Hard rule:** It is **not** a second Mech OS kernel. It **never** writes `AdmissionStatus`.

**v1:** local CAS + SQLite. CLI: `ed ingest` / `query` / `gaps` / `report` / `provenance` / `pack`.

**Fixtures:** material fixtures in v1 are **synthetic** — not AISI allowables. Verification with Mech OS keeps the boundary clear.

Mech OS Datastack answers: *what claims do we hold, with what trail and units?*

---

## 3) Reality OS — last gate before anyone confuses SIM with metal

**Job:** Fail-closed last-gate for Reality OS + Governor.

**Description string:** *Reality OS + Governor: fail-closed certified command path. SIM ≠ metal.*

**Governor:** may only allow, narrow, or abort. Cannot invent a plan or upgrade a refuse. Learned systems never write motors.

**Surfaces:** kernel, physics, data, plant, governor, core, session, ROS 2, viewport, HIL, plus a governor CLI.

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

Reality OS answers: *given a proposal, what is the governor allowed to do?*

---

## How a diligence conversation usually flows

1. Try **RoboSynx** — can we get a sane robot file into Isaac / MuJoCo / ROS 2 without install tax?
2. Try **SimTooReal** — can we see training failure modes and a transfer score before hardware?
3. Ask about **Mech OS** — how are physical claims unit-checked and admitted?
4. Ask about **Mech OS Datastack** — where do mechanical claims and provenance live?
5. Ask about **Reality OS** — what happens when a learned policy wants motor authority?

Skipping from step 1 to metal is how teams create demos that cannot survive a safety or procurement review.

## What we will not say

- That Mech OS or Mech OS Datastack are public GitHub products today (they are private R&D / pilot diligence unless we publish docs)
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
