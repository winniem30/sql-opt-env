#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sql_opt_env import SQLOptEnv, SQLOptAction

def test_basic():
    print("Testing basic functionality...")

    # Test reset
    env = SQLOptEnv()
    obs = env.reset()
    print(f"Reset OK: task={obs.task_id}, step={obs.step_number}")

    # Test step
    action = SQLOptAction(query="SELECT id, username FROM users WHERE is_active = TRUE")
    obs, reward, done, info = env.step(action)
    print(f"Step OK: reward={reward:.3f}, done={done}")

    # Test tasks
    from sql_opt_env.tasks import TASKS
    print(f"Tasks: {list(TASKS.keys())}")

    print("All tests passed!")

if __name__ == "__main__":
    test_basic()