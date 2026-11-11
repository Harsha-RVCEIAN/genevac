---
trigger: always_on
---

# Architecture and Contract Rules

## Integration boundary

- Person B consumes Pavan's simulation engine through its verified public interface.
- Analytics ingestion must be read-only. It must not advance, reset, or mutate the simulation.
- Verify actual method names, snapshot types, fields, timestamp/step semantics, coordinate frame/units, pedestrian identity/status, and scenario interface from current source and tests.
- Do not infer API contracts from old notes, mock-only tests, or guessed field names.
- If a required field/interface is missing, document the gap and ask for approval; do not independently modify Pavan's engine.

## Data identity and time

- Preserve pedestrian, run, snapshot, scenario, and experiment identity wherever available.
- Never mix trajectory histories across pedestrians or simulation runs.
- Keep observed, predicted, and ground-truth positions distinct.
- Never append predictions as observations.
- Document timestamp units/origin and coordinate frame/axes/units.
- Do not treat step index as elapsed time unless verified.
- Handle duplicate, out-of-order, missing, and invalid observations explicitly; never silently fabricate values.

## Contracts

Before implementing an integration, produce a field mapping from the actual engine output to the existing canonical data contract. Prefer a narrow adapter over changing either side's core schema. Preserve public APIs unless an approved requirement requires change.
