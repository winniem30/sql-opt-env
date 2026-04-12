"""
server.py — SQL Query Optimization OpenEnv
HF Space: winniem30/sql-open-env
Fixed: handles empty body {}, correct env API, no TASKS import
"""

import os
import json
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sql_opt_env import SQLOptEnv, SQLOptAction
from sql_opt_env.tasks import TASKS

app = FastAPI(title="SQL Query Optimization OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[SQLOptEnv] = None

def get_env() -> SQLOptEnv:
    global _env
    if _env is None:
        _env = SQLOptEnv("select_star_cleanup")
    return _env


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SQL Optimization OpenEnv is running.",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "sql_opt_env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": tid,
                "difficulty": t.difficulty,
                "description": t.description,
            }
            for tid, t in TASKS.items()
        ]
    }


@app.post("/reset")
async def reset(request: Request):
    """
    Reset endpoint — reads raw body to avoid Pydantic 422 on empty {}.
    The validator sends POST /reset with body: {} expecting HTTP 200.
    """
    global _env
    task_id = "select_star_cleanup"
    try:
        raw = await request.body()
        if raw and raw.strip() not in (b"", b"{}"):
            body = json.loads(raw)
            if isinstance(body, dict) and body.get("task_id"):
                tid = body["task_id"]
                if tid in TASKS:
                    task_id = tid
    except Exception:
        pass

    _env = SQLOptEnv(task_id=task_id)
    result = _env.reset()
    out = result.model_dump()
    out["task_id"] = task_id
    out["status"] = "ok"
    return JSONResponse(content=out)


@app.post("/step")
async def step(request: Request):
    query = ""
    explanation = None
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw)
            query = body.get("query", "")
            explanation = body.get("explanation")
    except Exception:
        pass

    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": "query field is required", "status": "error"}
        )

    result = get_env().step(SQLOptAction(query=query, explanation=explanation))
    out = result.model_dump()
    out["status"] = "ok"
    return JSONResponse(content=out)


@app.get("/state")
def state():
    out = get_env().state().model_dump()
    out["status"] = "ok"
    return JSONResponse(content=out)
def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    print(f"[server.app] SQL Opt Env v2.0.0 starting on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == '__main__':
    main()