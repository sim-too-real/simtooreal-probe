# PolicyTraceEvent schema (v1)

`SCHEMA_VERSION = 1` in `simtooreal_probe.trace`.

## Envelope fields

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | Always `1` for this package generation |
| `run_id` | str | Logical training / eval run |
| `grain` | `"iteration"` \| `"episode"` \| `"step"` | Discriminator |
| `source` | str | e.g. `sim:physx`, `sim:mujoco`, `real:<serial>` |
| `ts` | float | Wall time (seconds) |
| `iteration` | int \| null | Train iteration when known |
| `episode_id` | str \| null | Episode key at episode/step grains |
| `step_idx` | int \| null | Frame index within episode |
| `env_idx` | int \| null | Parallel env index when applicable |
| `scalars` | object | Named floats (reward, KL, …) |
| `vectors` | object | Named float arrays (`obs`, `action`, `reward_terms`, `policy_state`) |
| `tags` | object | Non-numeric labels (`success`, `termination`, …) |

## Canonical sources

- `sim:physx`, `sim:newton`, `sim:mjx`, `sim:mujoco`, `sim:mjwarp`
- `real:<serial>` via `real_source(serial)`

## Dim registry

`_dims.json` maps vector names → component name lists so notebooks can resolve
anonymous float indices to reward terms / joint names.

## Capture levels

| Level | Writes |
|-------|--------|
| `off` | nothing |
| `metrics` | iteration |
| `episodes` | iteration + episode |
| `full` | iteration + episode + step |
