# ErrorReductionPulseSync

Schedule-linked error model built on top of Qiskit circuits and explicit noise
assumptions.

This repo now states its scope directly:

- `simulation_baseline`: compare modeled random vs phase-locked schedules
- `hardware_derived_model`: anchor the same comparison to calibration-style
  device parameters
- `real_backend_validation`: pending

## What It Tests

Whether a phase-locked schedule outperforms a random schedule under the chosen
noise model and whether that result remains directionally stable once the model
is anchored to measured device characteristics.

Within the wider stack, this is one of the strongest transition-cadence repos.
It documents the narrower claim that cadence-linked synchronization matters
under the modeled conditions and gives the broader quantum-consciousness
architecture a concrete support layer.

## Quick Start

```bash
python3 error_reduction_experiment.py --mode simulation --json
python3 error_reduction_experiment.py --mode hardware-derived --json
```

See [docs/METHOD.md](docs/METHOD.md) and
[docs/EVIDENCE_BOUNDARY.md](docs/EVIDENCE_BOUNDARY.md).
