from typing import Any, Dict, Optional, Tuple
from random import choice

from .models import SQLOptAction, SQLOptObservation, SQLOptReward
from .tasks import TASKS, simulate_cost


def _clamp(value: float) -> float:
    """Clamp reward/score to strictly (0, 1) — never 0.0 or 1.0 exactly."""
    return round(min(max(float(value), 0.01), 0.99), 4)


class SQLOptEnv:
    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id
        self.task = None
        self.current_query = ""
        self.previous_query = ""
        self.steps = 0
        self.max_steps = 0
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.current_cost = 0.0

    def reset(self, task_id: Optional[str] = None) -> SQLOptObservation:
        if task_id is None:
            task_id = self.task_id if self.task_id else choice(list(TASKS.keys()))

        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.task = TASKS[task_id]
        self.current_query = self.task.original_query
        self.previous_query = self.current_query
        self.steps = 0
        self.max_steps = self.task.max_steps
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.current_cost = simulate_cost(self.current_query, self.task)

        return self._build_observation()

    def step(self, action: SQLOptAction) -> Tuple[SQLOptObservation, float, bool, Dict[str, Any]]:
        if self.task is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self.steps += 1
        candidate_query = action.query.strip() or self.current_query
        self.previous_query = self.current_query

        execution_error = None
        syntax_valid = candidate_query.upper().startswith(("SELECT", "WITH"))
        destructive = any(keyword in candidate_query.upper() for keyword in ["DELETE", "DROP", "TRUNCATE", "ALTER"])

        if destructive:
            execution_error = "Destructive or non-query action rejected."
            reward = _clamp(-0.05)   # FIX: was -0.05 (could be out of range)
            done = True
            info = {
                "cost_improvement": 0.0,
                "grade_score": 0.0,
                "execution_error": execution_error,
                "syntax_valid": False,
            }
            return self._build_observation(execution_error=execution_error, syntax_valid=False), reward, done, info

        if not syntax_valid:
            execution_error = "Query must start with SELECT or WITH."
            reward = _clamp(-0.05)   # FIX: was -0.05 (could be out of range)
            done = self.steps >= self.max_steps
            info = {
                "cost_improvement": 0.0,
                "grade_score": 0.0,
                "execution_error": execution_error,
                "syntax_valid": False,
            }
            self.current_query = candidate_query
            return self._build_observation(execution_error=execution_error, syntax_valid=False), reward, done, info

        grade_score = self.task.grade(candidate_query)["score"]
        old_cost = self.current_cost
        new_cost = simulate_cost(candidate_query, self.task)
        cost_improvement = max(0.0, old_cost - new_cost)

        progress_bonus = max(0.0, grade_score - self.task.grade(self.current_query)["score"])
        cost_component = min(max(cost_improvement / max(1.0, self.task.original_cost), 0.0), 1.0)
        reward = 0.6 * grade_score + 0.3 * cost_component + 0.1 * progress_bonus

        if candidate_query == self.current_query:
            reward = reward - 0.02

        # FIX: was min(max(reward, 0.0), 1.0) which allowed exactly 0.0 and 1.0
        reward = _clamp(reward)

        self.current_query = candidate_query
        self.current_cost = new_cost
        self.last_reward = reward
        self.cumulative_reward += reward

        done = self.steps >= self.max_steps or grade_score >= self.task.success_threshold
        info = {
            "cost_improvement": round(cost_improvement, 2),
            "grade_score": round(_clamp(grade_score), 4),
            "execution_error": execution_error,
            "syntax_valid": syntax_valid,
        }

        return self._build_observation(), reward, done, info

    def state(self) -> str:
        return self.current_query

    def close(self) -> None:
        return None

    def _build_observation(self, execution_error: Optional[str] = None, syntax_valid: Optional[bool] = None) -> SQLOptObservation:
        syntax_valid = self.current_query.upper().startswith(("SELECT", "WITH")) if syntax_valid is None else syntax_valid
        current_cost = self.current_cost
        cost_ratio = round(self.task.original_cost / current_cost, 4) if current_cost > 0 else None
        issues = self.task.issues(self.current_query, self.previous_query)
        hints = self.task.hints(self.current_query, self.steps)

        return SQLOptObservation(
            task_id=self.task.id,
            difficulty=self.task.difficulty,
            description=self.task.description,
            schemas=self.task.schemas,
            original_query=self.task.original_query,
            current_query=self.current_query,
            step_number=self.steps,
            max_steps=self.max_steps,
            syntax_valid=syntax_valid,
            execution_error=execution_error,
            estimated_cost=current_cost,
            original_cost=self.task.original_cost,
            cost_improvement_ratio=cost_ratio,
            issues_found=issues,
            hints=hints,
            last_reward=self.last_reward,
            cumulative_reward=round(self.cumulative_reward, 4),
            query_plan=None,
            rows_returned=None,
        )