# RoboSynx + SimTooReal: a sincere stack for robot learning teams

Physical AI needs file honesty and training honesty.

This page is the captcha-free GitHub copy of that claim. It is a product narrative, not a safety certificate and not a substitute for on-robot evidence.

## The stack

- **[RoboSynx](https://robosynx.com)** — free URDF / SDF / MJCF generate, convert, validate, and 3D view. File honesty: inspect and convert the robot description you train against without a locked toolchain.
- **[SimTooReal](https://simtooreal.com)** — Isaac Lab / MuJoCo training ops, failure intelligence, and sim-to-real scoring. Training honesty: see where a policy fails and how far the sim is from metal.
- **Reality OS** — fail-closed last-gate design (`SIM ≠ metal`). This is an engineering posture: simulation is not the robot. **No ISO or SIL certificates are claimed.** Diligence write-up: [reality-os-rust](reality-os-rust.md).
- **Contact:** [vardhan@simtooreal.com](mailto:vardhan@simtooreal.com)

This repository (`simtooreal-probe`) is the open-core policy debugging layer: local-first capture and failure replay. It does not certify a policy or a robot for the factory floor.

## Deeper stack (diligence / pilot)

The public products stop at file honesty and training honesty. Diligence asks three harder questions: what physical knowledge is *admitted*, where mechanical claims live with units and provenance, and who may command anything near metal.

**First Principles**, **engineering-data**, and **Reality OS** are the answers we give in that conversation. They are **pilot / diligence layers**, not public open-source dumps and not a claim that SIM evidence is metal. Limits we will keep saying: **`SIM ≠ metal`**. **No ISO or SIL certificates. No metal MEASURED status.**

| Layer | Diligence role | Honest limit |
|-------|----------------|--------------|
| **[First Principles](first-principles.md)** | Unit-checked mechanisms and admission for physical claims | Private R&D / pilot diligence — not a public kernel dump; not ISO/SIL |
| **[engineering-data](engineering-data.md)** | Machine-usable mechanical claims with rights, provenance, and SI | Private R&D / pilot diligence — not a second admission kernel; v1 fixtures are synthetic |
| **[Reality OS](reality-os-rust.md)** | Fail-closed last-gate (`allow` / `narrow` / `abort` only) | `SIM ≠ metal`. No ISO/SIL, no metal MEASURED, no independent HW e-stop claimed |

The combined map: **[deeper stack](deeper-stack.md)**.

## Measuring the gap

A five-pillar framing for policy readiness:

[Measuring the Sim-to-Real Gap: a Five-Pillar Framework for Robot Policy Readiness](https://simtoorealhq.hashnode.dev/measuring-the-sim-to-real-gap-a-five-pillar-framework-for-robot-policy-readiness)
