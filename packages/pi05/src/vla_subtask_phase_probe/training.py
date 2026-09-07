from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .detector import ProbeParameters


@dataclass(frozen=True)
class TrainingResult:
    parameters: ProbeParameters
    validation_accuracy: float
    predictions: NDArray[np.int64]


def phase_labels(length: int, transition_frames: tuple[int, ...]) -> NDArray[np.int64]:
    if length < 1:
        raise ValueError("length must be positive")
    if any(left >= right for left, right in zip(transition_frames, transition_frames[1:])):
        raise ValueError("transition frames must be strictly increasing")
    if transition_frames and not 0 < transition_frames[0] < length:
        raise ValueError("transition frames must be inside the rollout")
    if transition_frames and transition_frames[-1] >= length:
        raise ValueError("transition frames must be inside the rollout")
    labels = np.zeros(length, dtype=np.int64)
    for phase_index, frame in enumerate(transition_frames, start=1):
        labels[frame:] = phase_index
    return labels


def train_linear_probe(
    actions: NDArray[np.floating],
    transition_frames: tuple[int, ...],
    *,
    regularization: float = 1e-4,
    validation_stride: int = 5,
) -> TrainingResult:
    """Fit the original experiment's softmax linear probe.

    PyTorch is imported lazily so inference users only need NumPy. Install the
    ``train`` extra before calling this function.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("training requires: uv sync --extra train") from exc

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("actions must have shape (frames, action_dim)")
    if validation_stride < 2:
        raise ValueError("validation_stride must be at least 2")
    labels = phase_labels(len(values), transition_frames)
    validation = np.arange(len(values)) % validation_stride == 0
    training = ~validation
    mean = values[training].mean(axis=0)
    scale = values[training].std(axis=0) + 1e-5
    normalized = (values - mean) / scale

    torch.manual_seed(0)
    model = torch.nn.Linear(values.shape[1], len(transition_frames) + 1)
    features = torch.from_numpy(normalized[training])
    targets = torch.from_numpy(labels[training])
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=300, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features), targets)
        loss += regularization * sum(
            parameter.square().sum() for parameter in model.parameters()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized)).argmax(dim=1).numpy()
    parameters = ProbeParameters(
        weight=model.weight.detach().numpy(),
        bias=model.bias.detach().numpy(),
        mean=mean,
        scale=scale,
    )
    return TrainingResult(
        parameters=parameters,
        validation_accuracy=float((predictions[validation] == labels[validation]).mean()),
        predictions=np.asarray(predictions, dtype=np.int64),
    )
