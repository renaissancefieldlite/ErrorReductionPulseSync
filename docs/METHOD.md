# Method

This repo is the schedule-comparison layer for cadence-linked error behavior.

The Qiskit circuits provide an ideal probability distribution. The repo then
applies explicit noise schedules to those probabilities:

- a random schedule
- a phase-locked schedule

In hardware-derived mode the schedule severities are anchored to calibration
parameters such as gate error, readout error, crosstalk, and drift.

That gives the project a concrete answer to a fair review objection:
calibration-anchored modeling is stronger than an unconstrained toy parameter,
while real backend measurement remains the next stronger lane.
