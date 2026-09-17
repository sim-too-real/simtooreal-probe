# RoboSynx + SimTooReal: a sincere stack for robot learning teams

Physical AI needs file honesty and training honesty.

This page is the captcha-free GitHub copy of that claim. It is a product narrative, not a safety certificate and not a substitute for on-robot evidence.

## The stack

- **[RoboSynx](https://robosynx.com)** — free URDF / SDF / MJCF generate, convert, validate, and 3D view. File honesty: inspect and convert the robot description you train against without a locked toolchain.
- **[SimTooReal](https://simtooreal.com)** — Isaac Lab / MuJoCo training ops, failure intelligence, and sim-to-real scoring. Training honesty: see where a policy fails and how far the sim is from metal.
- **Reality OS** — fail-closed last-gate design (`SIM ≠ metal`). This is an engineering posture: simulation is not the robot. **No ISO or SIL certificates are claimed.**
- **Contact:** [vardhan@simtooreal.com](mailto:vardhan@simtooreal.com)

This repository (`simtooreal-probe`) is the open-core policy debugging layer: local-first capture and failure replay. It does not certify a policy or a robot for the factory floor.

## Measuring the gap

A five-pillar framing for policy readiness:

[Measuring the Sim-to-Real Gap: a Five-Pillar Framework for Robot Policy Readiness](https://simtoorealhq.hashnode.dev/measuring-the-sim-to-real-gap-a-five-pillar-framework-for-robot-policy-readiness)
