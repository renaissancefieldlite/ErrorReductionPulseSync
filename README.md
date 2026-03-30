# Experiment 4: ErrorReductionPulseSync

Schedule-linked error model for the `0.67 Hz` stack.

This repo now separates three layers:

1. `simulation_baseline`
   Compare modeled random versus phase-locked schedules under an explicit local
   error model.
2. `hardware_derived_model`
   Anchor the same comparison to calibration-style device parameters.
3. `real_backend_ordered_capture_validation`
   Compare the chronological FEZ backend trace against randomized reorderings of
   the same measured capture distributions.

## What It Tests

This repo asks one bounded question across three rungs:

- does a phase-linked or chronology-preserving schedule show lower transition
  error than a randomized schedule?

That question is tested first in a local model, then in a calibration-anchored
model, then directly on measured FEZ capture distributions.

Within the wider stack, this is one of the stronger transition-cadence repos.
It does not claim universal proof. It documents whether schedule structure
survives contact with more realistic and then real backend traces.

## Current Status

- simulation baseline: implemented
- hardware-derived model: implemented
- backend-capture ordering validation: implemented

## Current v3 Read

The current local `v3` outcome is mixed in the right way:

- simulation baseline supports the expected direction
  - `58.24%` relative error reduction
- hardware-derived model still supports the expected direction
  - `30.92%` relative error reduction
- the first real FEZ backend batch does not confirm the ordering advantage yet
  - one direct run: `0%`
  - 25-repeat shuffle summary: `-2.29%` mean relative error reduction

That means the modeled schedule-linked effect survives in simulation and
hardware-derived form, but the current `8`-capture FEZ set is still too weak or
too mixed to validate the same advantage cleanly on the real backend rung.

The repo keeps that result instead of smoothing it away. That is what makes the
`v3` layer useful.

## Quick Start

```bash
python3 error_reduction_experiment.py --mode simulation --json
python3 error_reduction_experiment.py --mode hardware-derived --json
python3 error_reduction_experiment.py \
  --mode backend-capture \
  --captures ../renaissancefieldlitehrv1.0/data/batches/ibm_ibm_fez_window_sweep_fixed500_acct2_1490ms_1774794095.json \
  --json
```

Repeated-run baseline:

```bash
python3 error_reduction_experiment.py --mode simulation --repeats 25 --json
python3 error_reduction_experiment.py --mode hardware-derived --repeats 25 --json
python3 error_reduction_experiment.py \
  --mode backend-capture \
  --captures ../renaissancefieldlitehrv1.0/data/batches/ibm_ibm_fez_window_sweep_fixed500_acct2_1490ms_1774794095.json \
  --repeats 25 \
  --json
```

Saved examples:

- [examples/latest_simulation_report.json](examples/latest_simulation_report.json)
- [examples/latest_hardware_report.json](examples/latest_hardware_report.json)
- [examples/latest_backend_report.json](examples/latest_backend_report.json)

See [docs/METHOD.md](docs/METHOD.md),
[docs/EVIDENCE_BOUNDARY.md](docs/EVIDENCE_BOUNDARY.md),
[docs/RETOOL_DIRECTION.md](docs/RETOOL_DIRECTION.md), and
[docs/PROPOSAL_RELEVANCE.md](docs/PROPOSAL_RELEVANCE.md).
