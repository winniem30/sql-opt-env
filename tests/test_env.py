from sql_opt_env.env import SQLOptEnv
from sql_opt_env.models import SQLOptAction


def test_env_reset():
    env = SQLOptEnv()
    obs = env.reset()

    assert obs.original_query is not None
    assert obs.step_number == 0


def test_env_step():
    env = SQLOptEnv()
    env.reset()

    action = SQLOptAction(
        query="SELECT id, name FROM users"
    )

    obs, reward, done, info = env.step(action)

    assert reward is not None
    assert 0.0 <= reward <= 1.0