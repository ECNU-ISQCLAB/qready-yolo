from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch

from nets.quantum_gate import (
    NUM_QUBITS,
    THETA_COUNT,
    QuantumChannelGate,
    QuantumGateStack,
    build_quantum_circuit,
)
from nets.yolo import YoloBody
from utils.utils import get_classes


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "show_the_circuit"
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
COCO_CLASSES_PATH = ROOT / "model_data" / "coco_classes.txt"
PHI = "l"
QUANTUM_TARGETS = ("p3", "p4", "p5")
QUANTUM_DEPTH = 2


def build_quantum_yolo() -> YoloBody:
    class_names, _ = get_classes(str(COCO_CLASSES_PATH))
    return YoloBody(
        ANCHORS_MASK,
        len(class_names),
        PHI,
        pretrained=False,
        use_quantum=True,
        quantum_targets=QUANTUM_TARGETS,
        quantum_depth=QUANTUM_DEPTH,
    )


def collect_quantum_modules(model: YoloBody) -> tuple[list[str], int]:
    lines: list[str] = []
    total_theta = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantumGateStack):
            lines.append(f"[{name}] QuantumGateStack(depth={len(module.layers)})")
        elif isinstance(module, QuantumChannelGate):
            theta_count = int(module.theta.numel())
            total_theta += theta_count
            lines.append(
                f"  - [{name}] QuantumChannelGate(theta_shape={tuple(module.theta.shape)}, theta_count={theta_count})"
            )
    return lines, total_theta


def sanitize_ascii(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The encoding gbk has a limited charset.*",
        category=RuntimeWarning,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_quantum_yolo()
    module_lines, total_theta = collect_quantum_modules(model)

    sample_patch = torch.linspace(-0.8, 0.8, 9)
    sample_theta = torch.linspace(0.0, 1.95, THETA_COUNT)
    circuit = build_quantum_circuit(sample_patch, sample_theta)
    circuit_text = sanitize_ascii(str(circuit.draw(output="text")))

    summary_lines = [
        "Quantum YOLO structure",
        f"num_qubits: {NUM_QUBITS}",
        f"theta_per_quantum_gate: {THETA_COUNT}",
        f"quantum_targets: {', '.join(QUANTUM_TARGETS)}",
        f"quantum_depth_per_target: {QUANTUM_DEPTH}",
        f"total_theta_count: {total_theta}",
        "",
        "Quantum modules:",
        *module_lines,
    ]
    summary_text = "\n".join(summary_lines) + "\n"

    summary_path = OUTPUT_DIR / "summary.txt"
    circuit_path = OUTPUT_DIR / "circuit_ascii.txt"
    meta_path = OUTPUT_DIR / "summary.json"

    summary_path.write_text(summary_text, encoding="utf-8")
    circuit_path.write_text(circuit_text + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "num_qubits": NUM_QUBITS,
                "theta_per_quantum_gate": THETA_COUNT,
                "quantum_targets": list(QUANTUM_TARGETS),
                "quantum_depth_per_target": QUANTUM_DEPTH,
                "total_theta_count": total_theta,
                "summary_file": str(summary_path),
                "circuit_file": str(circuit_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Quantum YOLO structure ready.")
    print(f"num_qubits: {NUM_QUBITS}")
    print(f"theta_per_quantum_gate: {THETA_COUNT}")
    print(f"total_theta_count: {total_theta}")
    print(f"summary_file: {summary_path}")
    print(f"circuit_file: {circuit_path}")


if __name__ == "__main__":
    main()
