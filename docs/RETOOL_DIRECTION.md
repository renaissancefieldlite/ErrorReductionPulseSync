# Retool Direction

Experiment 4 should now be read as a three-rung schedule-comparison repo:

1. `simulation_baseline`
2. `hardware_derived_model`
3. `real_backend_ordered_capture_validation`

The important correction is that the backend rung is not treated as if it were
the same thing as the simulation rung.

What the backend rung actually does:

- uses measured FEZ capture distributions
- preserves their real chronology
- compares that chronology to randomized reorderings of the same capture set

That gives the repo a real backend contact layer without overstating what the
captures prove.

## Next rung

The next improvement is not more rhetoric. It is:

- denser FEZ session traces
- more repeated ordering comparisons
- comparison across multiple session windows

That is how Experiment 4 moves from a single backend continuity layer into a
stronger empirical schedule-comparison lane.
