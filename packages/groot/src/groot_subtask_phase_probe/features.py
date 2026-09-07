from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch


def _tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Expected tensor-like module output, got {type(output)!r}")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1)
    return (values * weights).sum(dim=1) / denominator


@dataclass
class FeatureSnapshot:
    arrays: dict[str, np.ndarray]
    hook_latency_ms: float
    extraction_latency_ms: dict[str, float]
    snapshot_latency_ms: float
    shapes: dict[str, list[int]]
    token_counts: dict[str, int]
    denoise_calls: int


class GrootFeatureCapture:
    """Read-only forward hooks for GR00T N1.7.

    The hooks detach outputs and never replace module inputs or outputs. The last
    action-encoder and DiT calls correspond to the fourth/final denoise iteration.
    """

    def __init__(self, gr00t_policy: Any):
        self.model = gr00t_policy.model
        self.head = self.model.action_head
        self._handles = []
        self._hook_latency_s = 0.0
        self._backbone: dict[str, torch.Tensor] | None = None
        self._state: torch.Tensor | None = None
        self._action_iterations: list[torch.Tensor] = []
        self._dit_iterations: list[torch.Tensor] = []
        self._handles.extend(
            [
                self.model.backbone.register_forward_hook(self._capture_backbone),
                self.head.state_encoder.register_forward_hook(self._capture_state),
                self.head.action_encoder.register_forward_hook(self._capture_action),
                self.head.model.register_forward_hook(self._capture_dit),
            ]
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_step(self) -> None:
        self._hook_latency_s = 0.0
        self._backbone = None
        self._state = None
        self._action_iterations = []
        self._dit_iterations = []

    def _timed_store(self, callback) -> None:
        start = perf_counter()
        callback()
        self._hook_latency_s += perf_counter() - start

    def _capture_backbone(self, _module, _inputs, output) -> None:
        def store():
            self._backbone = {
                "features": output["backbone_features"].detach(),
                "attention": output["backbone_attention_mask"].detach(),
                "image": output["image_mask"].detach(),
            }

        self._timed_store(store)

    def _capture_state(self, _module, _inputs, output) -> None:
        self._timed_store(lambda: setattr(self, "_state", _tensor_from_output(output).detach()))

    def _capture_action(self, _module, _inputs, output) -> None:
        self._timed_store(
            lambda: self._action_iterations.append(_tensor_from_output(output).detach())
        )

    def _capture_dit(self, _module, _inputs, output) -> None:
        self._timed_store(lambda: self._dit_iterations.append(_tensor_from_output(output).detach()))

    @staticmethod
    def _cpu_vector(value: torch.Tensor) -> np.ndarray:
        return value[0].float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)

    def snapshot(self, raw_state: np.ndarray, decoded_action_chunk: np.ndarray) -> FeatureSnapshot:
        snapshot_started = perf_counter()
        if self._backbone is None or self._state is None:
            raise RuntimeError("Backbone/state hooks did not fire")
        expected_calls = int(self.head.num_inference_timesteps)
        if (
            len(self._action_iterations) != expected_calls
            or len(self._dit_iterations) != expected_calls
        ):
            raise RuntimeError(
                f"Expected {expected_calls} denoise calls, got action={len(self._action_iterations)} "
                f"dit={len(self._dit_iterations)}"
            )

        backbone = self._backbone["features"]
        attention = self._backbone["attention"].bool()
        image_mask = self._backbone["image"].bool() & attention
        instruction_mask = ~self._backbone["image"].bool() & attention
        if not bool(image_mask.any()):
            raise RuntimeError("Backbone output does not contain valid image tokens")
        if not bool(instruction_mask.any()):
            raise RuntimeError("Backbone output does not contain valid instruction tokens")
        action_last = self._action_iterations[-1]
        dit_full = self._dit_iterations[-1]
        horizon = int(self.head.action_horizon)
        dit_action = dit_full[:, -horizon:, :]

        extraction_latency_ms: dict[str, float] = {}

        def extract(name: str, callback) -> np.ndarray:
            started = perf_counter()
            value = callback()
            extraction_latency_ms[name] = (perf_counter() - started) * 1000
            return value

        arrays: dict[str, np.ndarray] = {}
        arrays["R0_raw_state"] = extract(
            "R0_raw_state",
            lambda: np.asarray(raw_state, dtype=np.float32).reshape(-1),
        )
        arrays["R1_action_chunk"] = extract(
            "R1_action_chunk",
            lambda: np.asarray(decoded_action_chunk, dtype=np.float32).reshape(-1),
        )
        arrays["R2_state_encoder"] = extract(
            "R2_state_encoder", lambda: self._cpu_vector(self._state)
        )
        arrays["R3_vlm_hidden"] = extract(
            "R3_vlm_hidden",
            lambda: self._cpu_vector(_masked_mean(backbone, attention)),
        )
        arrays["R4_action_encoder"] = extract(
            "R4_action_encoder",
            lambda: self._cpu_vector(action_last.mean(dim=1)),
        )
        arrays["R5_final_dit_hidden"] = extract(
            "R5_final_dit_hidden",
            lambda: self._cpu_vector(dit_action.mean(dim=1)),
        )
        arrays["R6_vlm_dit_concat"] = extract(
            "R6_vlm_dit_concat",
            lambda: np.concatenate([arrays["R3_vlm_hidden"], arrays["R5_final_dit_hidden"]]).astype(
                np.float32, copy=False
            ),
        )
        arrays["R3b_vision_tokens"] = extract(
            "R3b_vision_tokens",
            lambda: self._cpu_vector(_masked_mean(backbone, image_mask)),
        )
        arrays["R3c_instruction_tokens"] = extract(
            "R3c_instruction_tokens",
            lambda: self._cpu_vector(_masked_mean(backbone, instruction_mask)),
        )
        arrays["F1_vision_instruction"] = extract(
            "F1_vision_instruction",
            lambda: np.concatenate(
                [arrays["R3b_vision_tokens"], arrays["R3c_instruction_tokens"]]
            ).astype(np.float32, copy=False),
        )

        shapes = {
            "backbone": list(backbone.shape),
            "state_encoder": list(self._state.shape),
            "action_encoder_last": list(action_last.shape),
            "dit_full_last": list(dit_full.shape),
            "dit_action_last": list(dit_action.shape),
        }
        token_counts = {
            "valid": int(attention.sum().item()),
            "vision": int(image_mask.sum().item()),
            "instruction": int(instruction_mask.sum().item()),
        }
        return FeatureSnapshot(
            arrays=arrays,
            hook_latency_ms=self._hook_latency_s * 1000,
            extraction_latency_ms=extraction_latency_ms,
            snapshot_latency_ms=(perf_counter() - snapshot_started) * 1000,
            shapes=shapes,
            token_counts=token_counts,
            denoise_calls=expected_calls,
        )
