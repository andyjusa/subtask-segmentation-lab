from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Pi05RepresentationCollector:
    """Capture PI0.5 action-expert and VLM features without patching LeRobot.

    PI0.5 is a flow-matching policy and has no discrete action tokens. The input
    to ``action_out_proj`` is the final action-expert hidden slot and is the
    closest same-model equivalent to an action-token hidden representation.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self._latest_expert_hidden: Any | None = None
        self._latest_vlm_hidden: Any | None = None
        self._expert_queue: deque[NDArray[np.float32]] = deque()
        self._vlm_queue: deque[NDArray[np.float32]] = deque()
        self._plan_queue: deque[int] = deque()
        self._plan_index = -1
        self._original_predict: Callable[..., Any] = policy.predict_action_chunk

        core = policy.model
        self._handles = [
            core.action_out_proj.register_forward_hook(self._capture_expert_hidden),
            core.paligemma_with_expert.paligemma.model.language_model.norm.register_forward_hook(
                self._capture_vlm_hidden
            ),
        ]
        policy.predict_action_chunk = self._predict_and_finalize

    def close(self) -> None:
        self.policy.predict_action_chunk = self._original_predict
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def consume_frame(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
        if not self._expert_queue or not self._vlm_queue or not self._plan_queue:
            raise RuntimeError("no PI0.5 representation is available for this action")
        return (
            self._expert_queue.popleft(),
            self._vlm_queue.popleft(),
            self._plan_queue.popleft(),
        )

    def _capture_expert_hidden(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del module, output
        self._latest_expert_hidden = inputs[0].detach()

    def _capture_vlm_hidden(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        self._latest_vlm_hidden = hidden.detach()

    def _predict_and_finalize(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        actions = self._original_predict(batch, **kwargs)
        if self._latest_expert_hidden is None or self._latest_vlm_hidden is None:
            raise RuntimeError("PI0.5 hooks did not observe the expected internal tensors")

        expert = self._latest_expert_hidden[0].float().cpu().numpy()
        pooled_vlm = self._pool_valid_prefix(self._latest_vlm_hidden, batch)
        steps = min(self.policy.config.n_action_steps, len(expert))
        self._plan_index += 1
        self._expert_queue.extend(np.asarray(expert[:steps], dtype=np.float32))
        self._vlm_queue.extend(
            np.repeat(pooled_vlm[None, :], steps, axis=0).astype(np.float32)
        )
        self._plan_queue.extend([self._plan_index] * steps)
        return actions

    def _pool_valid_prefix(
        self, hidden: Any, batch: dict[str, Any]
    ) -> NDArray[np.float32]:
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
        )

        values = hidden[0].float()
        language_mask = batch[OBS_LANGUAGE_ATTENTION_MASK][0].to(dtype=values.dtype)
        language_length = language_mask.numel()
        image_length = values.shape[0] - language_length
        image_keys = tuple(self.policy.config.image_features)
        if not image_keys or image_length % len(image_keys) != 0:
            raise RuntimeError("could not reconstruct PI0.5 prefix token mask")
        tokens_per_image = image_length // len(image_keys)
        present_images = sum(key in batch for key in image_keys)
        image_mask = values.new_zeros(image_length)
        image_mask[: present_images * tokens_per_image] = 1
        import torch

        valid = torch.cat([image_mask, language_mask])
        pooled = (values * valid[:, None]).sum(dim=0) / valid.sum().clamp_min(1)
        return np.asarray(pooled.cpu().numpy(), dtype=np.float32)
