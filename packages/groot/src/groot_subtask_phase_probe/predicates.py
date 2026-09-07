from typing import Any

import gymnasium as gym

from .tasks import TaskSpec


def _libero_bddl_env(env: gym.Env) -> Any:
    current: Any = env.unwrapped
    if not hasattr(current, "_env"):
        raise TypeError("Expected GR00T LiberoEnv with an _env attribute")
    current = current._env
    if hasattr(current, "env"):
        current = current.env
    if not hasattr(current, "_eval_predicate"):
        raise TypeError("Could not locate LIBERO BDDL predicate evaluator")
    return current


def evaluate_predicate_1(env: gym.Env, task: TaskSpec) -> bool:
    return bool(_libero_bddl_env(env)._eval_predicate(list(task.predicate_1)))


class PredicateInfoWrapper(gym.Wrapper):
    """Adds exact simulator predicates to every raw environment step."""

    def __init__(self, env: gym.Env, task: TaskSpec):
        super().__init__(env)
        self.task = task

    def _signals(self) -> dict[str, bool]:
        return {
            "probe_predicate_1": evaluate_predicate_1(self.env, self.task),
            "probe_task_success": bool(self.env.unwrapped._env.check_success()),
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, {**info, **self._signals()}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, {**info, **self._signals()}
