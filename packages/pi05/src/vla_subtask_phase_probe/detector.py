from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import TaskSignal


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ProbeParameters:
    """Linear probe parameters and action normalization statistics."""

    weight: FloatArray
    bias: FloatArray
    mean: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError("weight must have shape (phases, action_dim)")
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias does not match the phase count")
        if mean.shape != (weight.shape[1],) or scale.shape != mean.shape:
            raise ValueError("normalization does not match the action dimension")
        if np.any(scale <= 0):
            raise ValueError("all scale values must be positive")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def load(cls, checkpoint: Path | str) -> ProbeParameters:
        with np.load(checkpoint) as values:
            return cls(
                weight=values["weight"],
                bias=values["bias"],
                mean=values["mean"],
                scale=values["scale"],
            )

    def save(self, checkpoint: Path | str, phases: tuple[str, ...]) -> None:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weight=self.weight,
            bias=self.bias,
            mean=self.mean,
            scale=self.scale,
            phases=np.asarray(phases),
        )


class ActionPhaseProbeDetector:
    """Emit ordered phase transitions from one policy action per control frame.

    A transition is accepted only when the immediately following phase is the
    highest-probability class above ``confidence_threshold`` for
    ``stable_frames`` consecutive observations. Phases cannot be skipped or
    emitted twice.
    """

    def __init__(
        self,
        *,
        parameters: ProbeParameters,
        phases: tuple[str, ...],
        task_id: str,
        subtask_ids: tuple[str, ...],
        confidence_threshold: float = 0.5,
        stable_frames: int = 5,
    ) -> None:
        if len(phases) < 2:
            raise ValueError("at least two ordered phases are required")
        if parameters.weight.shape[0] != len(phases):
            raise ValueError("probe output count does not match phases")
        if len(subtask_ids) != len(phases):
            raise ValueError("one subtask ID is required for each phase")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        self.parameters = parameters
        self.phases = phases
        self.task_id = task_id
        self.subtask_ids = subtask_ids
        self.confidence_threshold = confidence_threshold
        self.stable_frames = stable_frames
        self.phase_index = 0
        self._started = False
        self._stable_count = 0

    @property
    def phase(self) -> str:
        return self.phases[self.phase_index]

    def start(self) -> TaskSignal:
        if self._started:
            raise RuntimeError("task has already started")
        self._started = True
        return self._signal()

    def probabilities(self, action: NDArray[np.floating]) -> FloatArray:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != self.parameters.mean.shape:
            raise ValueError(
                f"expected {self.parameters.mean.size} action values, "
                f"received {values.size}"
            )
        normalized = (values - self.parameters.mean) / self.parameters.scale
        logits = self.parameters.weight @ normalized + self.parameters.bias
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        return np.asarray(probabilities / probabilities.sum(), dtype=np.float32)

    def observe(self, action: NDArray[np.floating]) -> TaskSignal | None:
        if not self._started:
            raise RuntimeError("start() must be called before observe()")
        if self.phase_index == len(self.phases) - 1:
            return None

        next_index = self.phase_index + 1
        probabilities = self.probabilities(action)
        condition = (
            int(np.argmax(probabilities)) == next_index
            and probabilities[next_index] >= self.confidence_threshold
        )
        self._stable_count = self._stable_count + 1 if condition else 0
        if self._stable_count < self.stable_frames:
            return None

        self.phase_index = next_index
        self._stable_count = 0
        return self._signal()

    def _signal(self) -> TaskSignal:
        return TaskSignal(
            task_id=self.task_id,
            task_sub_id=self.subtask_ids[self.phase_index],
            phase=self.phase,
        )
