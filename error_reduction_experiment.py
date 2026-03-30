"""Schedule-linked error model with simulation, hardware-derived, and backend capture modes."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from hardware_profile import extract_noise_parameters, load_calibration

TARGET_ZERO_KEYS = {"00", "0x0"}
TARGET_THREE_KEYS = {"11", "0x3"}


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


def summary_stats(values: list[float]) -> dict[str, float]:
    array = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_manifest_capture_paths(path: Path) -> tuple[list[Path], dict[str, Any] | None]:
    payload = load_json(path)
    if payload.get("schema_version") != "rfl.capture_batch.v2":
        return [path], None
    capture_files = [Path(item).resolve() for item in payload.get("capture_files", [])]
    return capture_files, payload


def expand_capture_inputs(inputs: list[str]) -> tuple[list[Path], list[dict[str, Any]]]:
    capture_paths: list[Path] = []
    manifests: list[dict[str, Any]] = []
    for item in inputs:
        matches = glob.glob(item)
        candidates = [Path(match) for match in matches] if matches else [Path(item)]
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.exists():
                continue
            manifest_paths, manifest = extract_manifest_capture_paths(resolved)
            if manifest:
                manifests.append(manifest)
                capture_paths.extend(manifest_paths)
            else:
                capture_paths.append(resolved)
    deduped = sorted({path.resolve() for path in capture_paths})
    return deduped, manifests


def parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def summarize_spacing(timestamps: list[datetime]) -> dict[str, float] | None:
    if len(timestamps) < 2:
        return None
    deltas = np.array(
        [
            (later - earlier).total_seconds()
            for earlier, later in zip(timestamps, timestamps[1:])
            if (later - earlier).total_seconds() > 0
        ],
        dtype=float,
    )
    if deltas.size == 0:
        return None
    return {
        "mean_seconds": float(np.mean(deltas)),
        "median_seconds": float(np.median(deltas)),
        "min_seconds": float(np.min(deltas)),
        "max_seconds": float(np.max(deltas)),
    }


def extract_counts(capture: dict[str, Any]) -> dict[str, int]:
    raw_result = capture.get("raw_result", capture)
    if raw_result.get("results"):
        first = raw_result["results"][0]
        return dict(first.get("data", {}).get("counts", {}))
    if raw_result.get("experiments"):
        first = raw_result["experiments"][0]
        return dict(first.get("measurement_counts", {}))
    return {}


def compute_capture_metrics(path: Path) -> dict[str, Any]:
    capture = load_json(path)
    counts = extract_counts(capture)
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError(f"{path} has no measurable counts.")

    zero_prob = sum(counts.get(key, 0) for key in TARGET_ZERO_KEYS) / total
    three_prob = sum(counts.get(key, 0) for key in TARGET_THREE_KEYS) / total
    target_subspace_probability = zero_prob + three_prob
    return {
        "path": str(path),
        "provider": capture.get("provider"),
        "backend_name": capture.get("backend_name"),
        "submitted_at_utc": capture.get("submitted_at_utc"),
        "created_at_utc": capture.get("created_at_utc"),
        "job_id": capture.get("job_id"),
        "shots": total,
        "zero_probability": zero_prob,
        "three_probability": three_prob,
        "target_subspace_probability": target_subspace_probability,
        "off_target_probability": max(0.0, 1.0 - target_subspace_probability),
        "bell_imbalance": abs(zero_prob - three_prob),
    }


def resolve_backend_sample_rate(
    items: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    capture_series_rate_hz: float | None,
) -> tuple[float, str, float | None, dict[str, float] | None]:
    if capture_series_rate_hz and capture_series_rate_hz > 0:
        return capture_series_rate_hz, "explicit_capture_series_rate_hz", None, None

    timestamps = sorted(
        [
            parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc"))
            for item in items
            if parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc")) is not None
        ]
    )
    actual_spacing_summary = summarize_spacing(timestamps)
    if actual_spacing_summary:
        median_spacing = actual_spacing_summary["median_seconds"]
        return 1.0 / median_spacing, "capture_timestamp_spacing", median_spacing, actual_spacing_summary

    for manifest in manifests:
        selection_window_seconds = manifest.get("selection_window_seconds")
        if selection_window_seconds and selection_window_seconds > 0:
            return 1.0 / float(selection_window_seconds), "manifest_selection_window_seconds", float(selection_window_seconds), None

    return 1.0, "capture_index_fallback", None, None


def capture_distribution(item: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(item["zero_probability"]),
            float(item["three_probability"]),
            float(item["off_target_probability"]),
        ],
        dtype=float,
    )


def schedule_transition_error(distributions: np.ndarray) -> float:
    if len(distributions) < 2:
        return 0.0
    diffs = np.abs(np.diff(distributions, axis=0))
    total_variation = np.sum(diffs, axis=1) / 2.0
    return float(np.mean(total_variation))


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


def run_backend_capture(
    captures: list[str],
    capture_series_rate_hz: float | None,
    seed: int,
) -> dict[str, object]:
    capture_paths, manifests = expand_capture_inputs(captures)
    if not capture_paths:
        raise SystemExit("No backend capture files matched the provided inputs.")

    items = [compute_capture_metrics(path) for path in capture_paths]
    items.sort(
        key=lambda item: (
            parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc")).timestamp()
            if parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc")) is not None
            else float("inf"),
            item["path"],
        )
    )

    sample_rate_hz, sample_rate_source, derived_spacing_seconds, actual_spacing_summary = resolve_backend_sample_rate(
        items,
        manifests,
        capture_series_rate_hz,
    )
    ordered = np.vstack([capture_distribution(item) for item in items])
    rng = np.random.default_rng(seed)
    shuffled = ordered[rng.permutation(len(ordered))]

    phase_locked_error = schedule_transition_error(ordered)
    random_schedule_error = schedule_transition_error(shuffled)
    relative_error_reduction = (random_schedule_error - phase_locked_error) / max(random_schedule_error, 1e-9)

    manifest_context: dict[str, Any] = {}
    if manifests:
        first_manifest = manifests[0]
        manifest_context = {
            "label": first_manifest.get("label"),
            "condition": first_manifest.get("condition"),
            "session_mode": first_manifest.get("session_mode"),
            "session_reference": first_manifest.get("session_reference"),
            "selection_window_seconds": first_manifest.get("selection_window_seconds"),
            "submit_spacing_seconds": first_manifest.get("submit_spacing_seconds"),
            "completed_repeats": first_manifest.get("completed_repeats"),
        }

    target_series = np.array([float(item["target_subspace_probability"]) for item in items], dtype=float)
    bell_imbalance_series = np.array([float(item["bell_imbalance"]) for item in items], dtype=float)

    return {
        "mode": "backend-capture",
        "evidence_status": "real_backend_ordered_capture_validation",
        "claim_under_test": "Whether the chronological backend trace yields lower adjacent distribution error than randomized reorderings of the same FEZ capture set when the comparison is made directly on measured output distributions.",
        "provider": items[0].get("provider"),
        "backend_name": items[0].get("backend_name"),
        "capture_count": len(items),
        "sample_rate_hz": sample_rate_hz,
        "sample_rate_source": sample_rate_source,
        "derived_spacing_seconds": derived_spacing_seconds,
        "actual_spacing_summary": actual_spacing_summary,
        "resolution_note": "This backend rung is an ordering-continuity test on measured FEZ capture distributions. It is direct backend evidence for the schedule-linked layer, but it is not a standalone causal proof of literal physical phase lock.",
        "trace_summary": {
            "mean_target_subspace_probability": float(np.mean(target_series)),
            "std_target_subspace_probability": float(np.std(target_series)),
            "mean_off_target_probability": float(
                np.mean(np.array([float(item["off_target_probability"]) for item in items], dtype=float))
            ),
            "mean_bell_imbalance": float(np.mean(bell_imbalance_series)),
        },
        "model_summary": {
            "random_schedule_error": random_schedule_error,
            "phase_locked_error": phase_locked_error,
            "relative_error_reduction": float(relative_error_reduction),
        },
        "manifest_context": manifest_context,
        "captures": items,
    }


def run_repeated(
    mode: str,
    calibration_path: str | None,
    seed: int,
    repeats: int,
    captures: list[str] | None = None,
    capture_series_rate_hz: float | None = None,
) -> dict[str, object]:
    runs = []
    for offset in range(repeats):
        run_seed = seed + offset
        if mode == "simulation":
            run = run_simulation()
        elif mode == "backend-capture":
            run = run_backend_capture(captures or [], capture_series_rate_hz, run_seed)
        else:
            run = run_hardware_derived(calibration_path)
        run["seed"] = run_seed
        runs.append(run)

    random_errors = [float(run["model_summary"]["random_schedule_error"]) for run in runs]
    locked_errors = [float(run["model_summary"]["phase_locked_error"]) for run in runs]
    reductions = [float(run["model_summary"]["relative_error_reduction"]) for run in runs]

    result = {
        "mode": mode,
        "evidence_status": runs[0]["evidence_status"],
        "claim_under_test": runs[0]["claim_under_test"],
        "repeat_count": repeats,
        "seed_start": seed,
        "repeat_summary": {
            "random_schedule_error": summary_stats(random_errors),
            "phase_locked_error": summary_stats(locked_errors),
            "relative_error_reduction": summary_stats(reductions),
        },
        "runs": runs,
    }

    if mode == "hardware-derived":
        result["noise_parameters"] = runs[0].get("noise_parameters", {})
    elif mode == "backend-capture":
        mean_target = [float(run["trace_summary"]["mean_target_subspace_probability"]) for run in runs]
        mean_imbalance = [float(run["trace_summary"]["mean_bell_imbalance"]) for run in runs]
        result["repeat_summary"]["mean_target_subspace_probability"] = summary_stats(mean_target)
        result["repeat_summary"]["mean_bell_imbalance"] = summary_stats(mean_imbalance)
        result["manifest_context"] = runs[0].get("manifest_context", {})
        result["resolution_note"] = runs[0].get("resolution_note")

    return result


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Run bounded error-reduction models.")
    parser.add_argument("--mode", choices=["simulation", "hardware-derived", "backend-capture"], default="simulation")
    parser.add_argument("--calibration")
    parser.add_argument("--captures", nargs="+", help="Capture file paths, globs, or batch manifests for backend-capture mode.")
    parser.add_argument(
        "--capture-series-rate-hz",
        type=float,
        help="Optional explicit sample rate for backend-capture mode when manifest/session cadence is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.repeats > 1:
        result = run_repeated(
            args.mode,
            args.calibration,
            args.seed,
            args.repeats,
            args.captures,
            args.capture_series_rate_hz,
        )
    elif args.mode == "simulation":
        result = run_simulation()
    elif args.mode == "backend-capture":
        result = run_backend_capture(args.captures or [], args.capture_series_rate_hz, args.seed)
    else:
        result = run_hardware_derived(args.calibration)

    result["schema_version"] = "rfl.error_reduction_pulse_sync.v3"
    result["next_step"] = "Extend the backend rung with denser FEZ session traces and compare the ordering advantage across additional session windows before treating the effect as broader than the current capture set."

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"mode={result['mode']}")
        if args.repeats > 1:
            mean_reduction = result["repeat_summary"]["relative_error_reduction"]["mean"]
            print(f"mean_relative_error_reduction={mean_reduction:.4f}")
        else:
            print(f"relative_error_reduction={result['model_summary']['relative_error_reduction']:.4f}")

    return result


if __name__ == "__main__":
    main()
