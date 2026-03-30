# Method

Experiment 4 is a schedule-comparison repo. It asks whether an ordered or
phase-linked schedule shows lower transition error than a randomized schedule
as the evidence layer becomes more realistic.

## Layer 1: Simulation baseline

The repo builds an ideal probability distribution from a bounded Qiskit circuit
and then applies two different schedule severities:

- a random schedule
- a phase-locked schedule

The comparison metric is the resulting error relative to the ideal state.

## Layer 2: Hardware-derived model

The same comparison is rerun, but the severity terms are no longer arbitrary.
They are anchored to calibration-style parameters such as:

- gate error
- readout error
- crosstalk
- `T1`
- `T2`

This is stronger than an unconstrained toy noise model, but it is still not
direct backend evidence.

## Layer 3: Backend capture ordering validation

The FEZ backend rung no longer injects a target in simulation. Instead, it
loads repeated backend capture files or a batch manifest, extracts the measured
output distributions, and compares:

- the real chronological order of those distributions
- a randomized reordering of the same measured distributions

The comparison metric is the adjacent total-variation transition error across
the ordered capture series.

That keeps the claim narrow:

- this layer is testing schedule/ordering continuity on measured backend output
- it is not claiming a standalone causal proof of literal physical phase lock

## Current v3 result

The current `v3` read should be stated plainly:

- the simulation and hardware-derived layers support the expected direction
- the current `8`-capture FEZ backend batch does not reproduce that ordering
  advantage cleanly
- the first direct backend comparison came back neutral
- the 25-repeat shuffle summary leaned slightly negative on average

That outcome is kept intentionally. The point of the backend rung is to force
contact with measured output, not to preserve the model result at all costs.
