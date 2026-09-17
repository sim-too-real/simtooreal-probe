# First Principles — unit-checked mechanisms as the unit of knowing

Physical AI fails quietly when teams treat English prose, spreadsheet cells, and bare floats as if they were physics. A mass written without a unit, a “verified” claim with no test, or a viewport that looks correct while the admission layer never ran — those are not edge cases. They are the default failure mode.

**First Principles** is our growing source of truth for admitted definitions, conservation laws, unit-checked equations, and experiments that either passed or failed. It is private R&D today — available for pilot diligence, not a public marketplace product. This article explains the design we are willing to stand behind in public.

Contact: **vardhan@simtooreal.com** · [simtooreal.com](https://simtooreal.com) · [robosynx.com](https://robosynx.com)

## The unit of knowing is a mechanism

In First Principles, knowledge is not a paragraph. The unit of knowing is a **mechanism**:

- **inputs**
- **outputs**
- **invariants**
- **test**

Mechanisms live under `mechanisms/`. English is an interface language only — useful for humans, never how truth is stored. If a claim cannot be expressed as a mechanism with a test, it is not admitted knowledge.

That sounds pedantic. It is deliberate. Robotics and industrial AI inherit centuries of unit mistakes; refusing bare numbers is cheaper than debugging them after a sim looks “fine.”

## Rust kernel: SI quantities, SI 2019 constants, honesty stamp

The `kernel/` is written in Rust. It provides:

- **SI `Qty` / `Dim`** — quantities carry dimensions; bare numbers without units are refused
- **SI 2019 constants** — the modern constant set, not a vibes-based physics layer
- **honesty stamp** — admission is an explicit status, not an implied vibe from a green viewport
- **mechanism parser** — mechanisms are parsed and checked, not free-text pasted into a notebook

The refusal of unitless bare numbers is a product decision. If your pipeline accepts `9.81` without saying whether that is m/s², the pipeline is lying by omission.

## Domains form a reduction stack

Admitted domains include:

| Layer | Domain |
|-------|--------|
| Foundations | `math` |
| Physical layers | `phys`, `chem`, `bio` |
| Meta / process | `fp` |
| Machines | `machine` |

This is a **reduction stack**. You cannot skip layers. A machine claim that pretends chemistry and physics do not apply is not admitted. That constraint is the point: physical AI that jumps straight from a prompt to a “machine” is marketing, not engineering.

## `machine::verify` is the sole simulation-admission authority

Simulation admission is not decided by what the viewport draws.

- **`machine::verify`** is the sole simulation-admission authority
- A FreeCAD add-on can call the First Principles workbench; the **viewport never admits**
- There is **no executable G-code** path from this stack
- Path / CAM capability is **named-missing** — we name the hole instead of implying a CAM product we do not ship

If something looks correct on screen but never passed `machine::verify`, it is not admitted. That is the honesty contract.

## Official vertical and program step

The official vertical we commit to in diligence is narrow on purpose:

- **Unpowered pinned structural linkage** — the class of machine we treat as in-scope for serious admission work
- **Program step:** one real physical lifecycle — not a slide deck of infinite machines

We do not claim to invent arbitrary machines end-to-end. We do not claim metal MEASURED status for things that only lived in SIM. Limits are part of the product surface.

## What First Principles is not

- Not a public open-source dump of the full kernel (today: private R&D / pilot diligence)
- Not a certified safety PLC or ISO/SIL certificate
- Not a second brain that invents plans and upgrades refusals into permissions
- Not “English in, any machine out”

## How it relates to RoboSynx and SimTooReal

[RoboSynx](https://robosynx.com) is the free developer surface for robot files (URDF / SDF / MJCF, validation, viewing, ROS 2 scaffold). [SimTooReal](https://simtooreal.com) is training ops and sim-to-real scoring for Isaac Lab and MuJoCo.

First Principles sits underneath as a **credibility layer**: when a physical claim needs unit-checked admission and a fail-closed story for diligence, this is the machinery. It is not the homepage CTA. It is what serious buyers ask about after they try the public tools.

## Contact

For diligence questions, pilots, or a technical walkthrough of the mechanism model:

**vardhan@simtooreal.com**

- Product: [simtooreal.com](https://simtooreal.com)
- Free robot-file tools: [robosynx.com](https://robosynx.com)

We would rather say what is refused than pretend a viewport is a proof.
