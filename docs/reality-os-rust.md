# reality-os-rust — Rust last-gate for Reality OS + Governor

Learned policies are good at proposing. They are dangerous when given unconstrained authority over motors. The industry often collapses “it worked in sim” into “it is safe on metal.” Those are different worlds.

**reality-os-rust** is the Rust last-gate for **Reality OS + Governor**: a fail-closed certified *command path* design (design intent — not an ISO/SIL certificate we pretend to hold). Public description string we use:

> Rust last-gate for Reality OS + Governor: fail-closed certified command path. SIM ≠ metal.

The repository is private on GitHub (`sim-too-real/reality-os-rust`), licensed **MIT**, and available for pilot diligence. It is a `theworld-runtime` port of the authority kernel.

Contact: **vardhan@simtooreal.com** · [simtooreal.com](https://simtooreal.com) · [robosynx.com](https://robosynx.com)

## Fail-closed Governor rules

The Governor may only:

- **allow**
- **narrow**
- **abort**

It **cannot**:

- invent a plan
- upgrade a refuse into an allow
- pretend SIM evidence is metal authority

**SIM ≠ metal** is stamped into the honesty model, not left as a slide footnote. Learned systems never write motors. That is not a branding line; it is an architectural refusal.

## Crate map

The workspace is split so authority, physics analogs, plant I/O, and session surfaces stay separable:

| Crate / surface | Role |
|-----------------|------|
| `realityos-kernel` | Core authority / last-gate types |
| `physics` | Physics-side support for gated paths |
| `data` | Data plane for gated sessions |
| `plant` | Plant interface boundary |
| `governor` | Allow / narrow / abort only |
| `core` | Shared primitives |
| `session` | Session lifecycle |
| `ros2` | ROS 2 integration surface |
| `vport` | Viewport / visualization boundary (does not grant metal authority) |
| `hil` | Hardware-in-the-loop analogs |
| `ros-governor` CLI | Operator-facing governor CLI |

The point of the split is auditability: a last-gate that is one opaque binary with “AI inside” is not something we want to defend in diligence.

## Honest ledger (what we claim and refuse)

We encourage public technical writing about this stack **only** when it stays inside the ledger:

| Claim area | Status we will say in public |
|------------|------------------------------|
| SIM last-gate | **Yes** — design and implementation intent for sim-side gated command path |
| ONLINE metal / MEASURED PFL | **No** |
| ISO 13850 / 10218 / 26262 / SIL | **No** |
| Independent HW e-stop | **No** (we do not claim we supply one) |
| Invent any machine | **No** |
| VLA last-write on actuators | **No** |

If a partner needs metal MEASURED, ISO/SIL, or independent hardware e-stop, that is a real engineering program — not something we paper over with a README badge.

## What “certified command path” means here

In our wording, **certified command path** means: the software path is designed so only allow / narrow / abort exits exist, refusals cannot be upgraded by the learner, and SIM is not silently re-labeled as metal.

It does **not** mean: third-party ISO 10218 / 26262 / SIL certification complete. When we do not have that evidence, we say **No**.

## How it sits next to RoboSynx and SimTooReal

- **[RoboSynx](https://robosynx.com)** — free browser tools for URDF / SDF / MJCF, validation, 3D view, ROS 2 scaffold
- **[SimTooReal](https://simtooreal.com)** — Isaac Lab / MuJoCo training ops, failure detection, transfer scoring
- **Reality OS / reality-os-rust** — last-gate when a team is ready to talk about command authority near hardware (pilot)

Typical path: clean files → honest training metrics → transfer score → only then a fail-closed gate conversation. Skipping to metal because a demo looked smooth is how trust dies.

## Pilots

For teams putting learned policies near metal, we discuss Reality OS as a **pilot upsell** with explicit scope: SIM/HIL analogs, governor behavior, and a written honesty ledger for what is still missing (metal MEASURED, independent e-stop, standards evidence).

**vardhan@simtooreal.com**

Fail closed. Name the holes. SIM ≠ metal.
