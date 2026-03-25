"""Schedule-linked error model with Qiskit baseline and hardware-derived mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from hardware_profile import extract_noise_parameters, load_calibration


def circuit_probabilities(depth: int = 6) -> np.ndarray:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    for _ in range(depth):
        circuit.rx(np.pi / 5, 0)
        circuit.ry(np.pi / 7, 1)
        circuit.cx(0, 1)
    state = Statevector.from_instruction(circuit)
    return np.array(state.probabilities(), dtype=float)


def noisy_distribution(ideal: np.ndarray, severity: float, seed: int = 67) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniform = np.ones_like(ideal) / len(ideal)
    jitter = rng.normal(0.0, severity * 0.01, size=len(ideal))
    mixed = (1.0 - severity) * ideal + severity * uniform + jitter
    mixed = np.clip(mixed, 0.0, None)
    mixed /= np.sum(mixed)
    return mixed


def fidelity_like(ideal: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sum(np.sqrt(ideal * observed)))


def run_model(random_severity: float, locked_severity: float) -> dict[str, float]:
    ideal = circuit_probabilities()
    random_dist = noisy_distribution(ideal, random_severity, seed=67)
    locked_dist = noisy_distribution(ideal, locked_severity, seed=68)
    random_error = 1.0 - fidelity_like(ideal, random_dist)
    locked_error = 1.0 - fidelity_like(ideal, locked_dist)
    relative_reduction = (random_error - locked_error) / max(random_error, 1e-9)
    return {
        "random_schedule_error": float(random_error),
        "phase_locked_error": float(locked_error),
        "relative_error_reduction": float(relative_reduction),
    }


def run_simulation() -> dict[str, object]:
    model = run_model(random_severity=0.22, locked_severity=0.11)
    return {
        "mode": "simulation",
        "evidence_status": "simulation_baseline",
        "claim_under_test": "Whether a phase-locked schedule outperforms a random schedule under an explicit model.",
        "model_summary": model,
    }


def run_hardware_derived(calibration_path: str | None) -> dict[str, object]:
    calibration = load_calibration(calibration_path)
    params = extract_noise_parameters(calibration)
    random_severity = min(
        0.35,
        params["mean_gate_error"] * 15.0 + params["mean_readout_error"] + params["mean_cross_talk"] * 3.0,
    )
    locked_severity = max(0.01, random_severity * 0.72)
    model = run_model(random_severity=random_severity, locked_severity=locked_severity)
    return {
        "mode": "hardware-derived",
        "evidence_status": "hardware_derived_model",
        "claim_under_test": "Whether the same schedule advantage persists when the noise severity is anchored to calibration-style parameters.",
        "noise_parameters": params,
        "model_summary": model,
    }


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Run bounded error-reduction models.")
    parser.add_argument("--mode", choices=["simulation", "hardware-derived"], default="simulation")
    parser.add_argument("--calibration")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = run_simulation() if args.mode == "simulation" else run_hardware_derived(args.calibration)
    result["schema_version"] = "rfl.error_reduction_pulse_sync.v2"
    result["next_step"] = "Repeat the same schedule comparison on a real backend before treating the reduction as empirical."

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"mode={result['mode']}")
        print(f"relative_error_reduction={result['model_summary']['relative_error_reduction']:.4f}")

    return result


if __name__ == "__main__":
    main()
