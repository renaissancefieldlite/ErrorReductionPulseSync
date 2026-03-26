# Method

This repo is a model-comparison layer, not direct hardware evidence.

The Qiskit circuits provide an ideal probability distribution. The repo then
applies explicit noise schedules to those probabilities:

- a random schedule
- a phase-locked schedule

In hardware-derived mode the schedule severities are anchored to calibration
parameters such as gate error, readout error, crosstalk, and drift.

That gives the project a concrete answer to a fair review objection:
calibration-anchored modeling is stronger than an unconstrained toy parameter,
but it is still not the same thing as a real backend measurement.
