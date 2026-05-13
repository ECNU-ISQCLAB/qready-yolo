import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


NUM_QUBITS = 3
THETA_COUNT = 40
REPLACE_THETA_COUNT = 9
_Z = SparsePauliOp.from_list([("Z", 1)])


def _encode_patch3x3(patch: torch.Tensor) -> QuantumCircuit:
    circuit = QuantumCircuit(NUM_QUBITS, name="QGateEnc3")
    for qubit in range(NUM_QUBITS):
        circuit.ry(float(patch[qubit]) * math.pi, qubit)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    for qubit in range(NUM_QUBITS):
        circuit.ry(float(patch[qubit + 3]) * math.pi, qubit)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    for qubit in range(NUM_QUBITS):
        circuit.ry(float(patch[qubit + 6]) * math.pi, qubit)
    return circuit


def build_quantum_circuit(patch: torch.Tensor, theta: torch.Tensor | list[float]) -> QuantumCircuit:
    if len(theta) != THETA_COUNT:
        raise ValueError(f"Expected theta length {THETA_COUNT}, got {len(theta)}.")

    circuit = _encode_patch3x3(patch)
    circuit.h(range(NUM_QUBITS))

    for idx, angle in enumerate(theta):
        qubit = idx % NUM_QUBITS
        entangle_step = idx % 3
        circuit.ry(float(angle), qubit)
        if entangle_step == 0:
            circuit.cx(0, 1)
        elif entangle_step == 1:
            circuit.cx(1, 2)
        else:
            circuit.cx(2, 0)
        if idx % 5 == 4:
            circuit.rz(float(angle) * 0.5, (qubit + 1) % NUM_QUBITS)
        if idx % 8 == 7:
            circuit.h((qubit + 2) % NUM_QUBITS)
    return circuit


def build_replace_quantum_circuit(
    patch: torch.Tensor,
    theta: torch.Tensor | list[float],
) -> QuantumCircuit:
    if len(theta) != REPLACE_THETA_COUNT:
        raise ValueError(f"Expected theta length {REPLACE_THETA_COUNT}, got {len(theta)}.")

    circuit = _encode_patch3x3(patch)
    for layer_idx in range(REPLACE_THETA_COUNT // NUM_QUBITS):
        offset = layer_idx * NUM_QUBITS
        for qubit in range(NUM_QUBITS):
            circuit.ry(float(theta[offset + qubit]), qubit)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 0)
    return circuit


def _quantum_expectation(patch: torch.Tensor, theta: torch.Tensor) -> float:
    circuit = build_quantum_circuit(patch, theta)
    state = Statevector.from_instruction(circuit)
    return float(state.expectation_value(_Z, qargs=[0]).real)


def _simulate_patch(patch: torch.Tensor, theta: torch.Tensor) -> tuple[float, list[float]]:
    center = _quantum_expectation(patch, theta)
    shift = math.pi / 2.0
    grads = []
    for idx in range(len(theta)):
        theta_plus = theta.clone()
        theta_minus = theta.clone()
        theta_plus[idx] += shift
        theta_minus[idx] -= shift
        grads.append(
            0.5
            * (
                _quantum_expectation(patch, theta_plus)
                - _quantum_expectation(patch, theta_minus)
            )
        )
    return center, grads


class _QuantumScalarFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, patches: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        theta_cpu = theta.detach().cpu()
        patch_rows = patches.detach().cpu()

        outputs = []
        grad_rows = []
        for patch in patch_rows:
            center, grads = _simulate_patch(patch, theta_cpu)
            outputs.append(center)
            grad_rows.append(grads)

        output_tensor = torch.tensor(outputs, dtype=patches.dtype, device=patches.device)
        grad_tensor = torch.tensor(grad_rows, dtype=patches.dtype)
        ctx.save_for_backward(grad_tensor)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_theta_matrix,) = ctx.saved_tensors
        grad_out_cpu = grad_output.detach().cpu()
        grad_theta = torch.matmul(grad_out_cpu.unsqueeze(0), grad_theta_matrix).squeeze(0)
        grad_theta = grad_theta.to(grad_output.device)
        grad_patches = None
        return grad_patches, grad_theta


class QuantumChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.theta = nn.Parameter(2 * math.pi * torch.rand(THETA_COUNT))
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1, eps=0.001, momentum=0.03),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.reduce(x)
        pooled = F.adaptive_avg_pool2d(pooled, (3, 3)).view(x.size(0), 9)
        pooled = torch.tanh(pooled)
        q_scalar = _QuantumScalarFunction.apply(pooled, self.theta).view(x.size(0), 1)
        gate = torch.tanh(self.fc2(F.leaky_relu(self.fc1(q_scalar), negative_slope=0.1)))
        gate = 1.0 + 0.1 * gate.view(x.size(0), x.size(1), 1, 1)
        return x * gate


class QuantumGateStack(nn.Module):
    def __init__(self, channels: int, depth: int = 2, reduction: int = 16):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        self.layers = nn.ModuleList(
            [QuantumChannelGate(channels, reduction=reduction) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _replace_quantum_expectations(patch: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    circuit = build_replace_quantum_circuit(patch, theta)
    state = Statevector.from_instruction(circuit)
    values = [
        float(state.expectation_value(_Z, qargs=[qubit]).real)
        for qubit in range(NUM_QUBITS)
    ]
    return torch.tensor(values, dtype=patch.dtype)


class QuantumReplaceBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta = nn.Parameter(2 * math.pi * torch.rand(REPLACE_THETA_COUNT))

    def _expand_channels(self, x: torch.Tensor) -> torch.Tensor:
        repeats = math.ceil(self.out_channels / self.in_channels)
        expanded = x.repeat(1, repeats, 1, 1)
        return expanded[:, : self.out_channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._expand_channels(x)
        pooled = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (3, 3)).view(x.size(0), 9)
        pooled = torch.tanh(pooled)

        exp_rows = []
        theta_cpu = self.theta.detach().cpu()
        for patch in pooled.detach().cpu():
            exp_rows.append(_replace_quantum_expectations(patch, theta_cpu))
        expectations = torch.stack(exp_rows).to(device=x.device, dtype=x.dtype)

        repeats = math.ceil(self.out_channels / NUM_QUBITS)
        channel_gate = expectations.repeat(1, repeats)[:, : self.out_channels]
        channel_gate = 1.0 + 0.2 * channel_gate.unsqueeze(-1).unsqueeze(-1)
        return base * channel_gate
