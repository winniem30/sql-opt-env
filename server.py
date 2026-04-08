from typing import Optional

from fastapi import FastAPI, Body, Request
from pydantic import BaseModel

from sql_opt_env import SQLOptAction, SQLOptEnv, TASKS

app = FastAPI(title="SQL Query Optimization OpenEnv")
env = SQLOptEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    query: str
    explanation: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SQL Optimization OpenEnv is running.",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    }


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": task.id,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
                "original_cost": task.original_cost,
            }
            for task in TASKS.values()
        ]
    }


@app.post("/reset")
def reset():
    observation = env.reset()
    return {"observation": observation.model_dump()}


@app.post("/step")
def step(payload: StepRequest):
    action = SQLOptAction(query=payload.query, explanation=payload.explanation)
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    return {"current_query": env.state()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)