"""
server/app.py - SQL Query Optimization OpenEnv
Entry point: server.app:main
"""

import os
import json
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="SQL Query Optimization OpenEnv", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASKS = {
    "select_star_cleanup": {
        "task_id": "select_star_cleanup",
        "difficulty": "easy",
        "description": (
            "Dashboard query uses SELECT DISTINCT * on 1M-row users table. "
            "Optimize: remove SELECT *, remove DISTINCT, add is_active filter, add LIMIT."
        ),
        "original_query": "SELECT DISTINCT *\nFROM users\nWHERE country = 'US'\nORDER BY created_at DESC;",
        "hints": [
            "Replace SELECT * with SELECT username, email",
            "Remove DISTINCT - user_id is PRIMARY KEY",
            "Add WHERE is_active = TRUE (indexed)",
            "Add LIMIT 100",
        ],
    },
    "n_plus_one_to_join": {
        "task_id": "n_plus_one_to_join",
        "difficulty": "medium",
        "description": (
            "Correlated subquery in SELECT runs once per user (N+1). "
            "Rewrite using JOIN + GROUP BY or CTE. Output: user_id, username, total_spent."
        ),
        "original_query": (
            "SELECT u.user_id, u.username,\n"
            "    (SELECT SUM(o.total_amount) FROM orders o\n"
            "     WHERE o.user_id = u.user_id AND o.status = 'completed') AS total_spent\n"
            "FROM users u WHERE u.is_active = TRUE;"
        ),
        "hints": [
            "Use LEFT JOIN orders ON user_id with GROUP BY",
            "Or: WITH totals AS (SELECT user_id, SUM(total_amount)...)",
            "Preserve status='completed' and is_active filters",
        ],
    },
    "window_function_rewrite": {
        "task_id": "window_function_rewrite",
        "difficulty": "hard",
        "description": (
            "Four nested correlated subqueries - O(N^2). "
            "Rewrite with ROW_NUMBER() OVER, SUM() OVER, and CTEs. "
            "Output: user_id, order_id, order_rank, order_total, running_total_spend. Top 3 per user."
        ),
        "original_query": (
            "SELECT u.user_id, o.order_id,\n"
            "    (SELECT COUNT(*) FROM orders o2 WHERE o2.user_id = o.user_id\n"
            "       AND o2.created_at >= o.created_at) AS order_rank,\n"
            "    (SELECT SUM(oi.unit_price * oi.quantity) FROM order_items oi\n"
            "     WHERE oi.order_id = o.order_id) AS order_total,\n"
            "    (SELECT SUM(o3.total_amount) FROM orders o3\n"
            "     WHERE o3.user_id = o.user_id\n"
            "       AND (SELECT COUNT(*) FROM orders o4 WHERE o4.user_id = o.user_id\n"
            "            AND o4.created_at >= o3.created_at) <= 3) AS running_total_spend\n"
            "FROM users u JOIN orders o ON u.user_id = o.user_id\n"
            "WHERE u.is_active = TRUE\n"
            "  AND (SELECT COUNT(*) FROM orders o5 WHERE o5.user_id = o.user_id\n"
            "       AND o5.created_at >= o.created_at) <= 3;"
        ),
        "hints": [
            "CTE order_totals: pre-aggregate order_items by order_id",
            "CTE ranked_orders: ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC)",
            "Filter WHERE order_rank <= 3",
            "SUM(order_total) OVER (PARTITION BY user_id ORDER BY order_rank) for running total",
        ],
    },
}


def _grade(task_id: str, query: str) -> dict:
    q = query.strip().upper()
    issues, passed, parts = [], [], {}
    if not (q.startswith("SELECT") or q.startswith("WITH")):
        return {"score": 0.01, "issues": ["Not a SELECT/WITH statement"], "passed_checks": []}

    if task_id == "select_star_cleanup":
        if "SELECT *" not in q:
            parts["a"] = 0.25; passed.append("Removed SELECT *")
        else:
            issues.append("Still using SELECT *")
        if "DISTINCT" not in q:
            parts["b"] = 0.15; passed.append("Removed DISTINCT")
        else:
            issues.append("DISTINCT unnecessary on PRIMARY KEY table")
        if "IS_ACTIVE" in q:
            parts["c"] = 0.20; passed.append("Filters on is_active")
        else:
            issues.append("Add WHERE is_active = TRUE (indexed)")
        if "COUNTRY" in q and ("'US'" in q or '"US"' in q):
            parts["d"] = 0.10; passed.append("Preserves country filter")
            if "USERNAME" in q and "EMAIL" in q:
                parts["e"] = 0.20; passed.append("Selects only username, email")
            else:
                issues.append("Select only username and email")
        else:
            issues.append("Preserve WHERE country = 'US'")
        if "LIMIT" in q:
            parts["f"] = 0.10; passed.append("Added LIMIT")
        else:
            issues.append("Add LIMIT for pagination")

    elif task_id == "n_plus_one_to_join":
        sc = q.count("SELECT")
        has_corr = sc >= 2 and "GROUP BY" not in q
        if sc == 1:
            parts["a"] = 0.30; passed.append("Eliminated correlated subquery")
        elif "WITH " in q and sc == 2 and not has_corr:
            parts["a"] = 0.20; passed.append("CTE eliminates N+1")
        else:
            issues.append("Correlated subquery still present")
        if "JOIN" in q or ("WITH " in q and "AS (" in q):
            parts["b"] = 0.20; passed.append("Uses JOIN or CTE")
        else:
            issues.append("Use JOIN or CTE")
        if "USER_ID" in q and "USERNAME" in q:
            parts["c"] = 0.15; passed.append("Returns user_id, username")
        else:
            issues.append("Must return user_id and username")
        if "'COMPLETED'" in q or '"COMPLETED"' in q:
            parts["d"] = 0.15; passed.append("Preserves status filter")
        else:
            issues.append("Missing status = 'completed'")
        if "IS_ACTIVE" in q:
            parts["e"] = 0.10; passed.append("Preserves is_active filter")
        else:
            issues.append("Missing is_active = TRUE")
        if "SUM(" in q:
            parts["f"] = 0.10; passed.append("Uses SUM aggregation")
        else:
            issues.append("Missing SUM for total_spent")

    elif task_id == "window_function_rewrite":
        if "OVER (" in q or "OVER(" in q:
            parts["a"] = 0.25; passed.append("Uses window functions")
        else:
            issues.append("Must use OVER clause")
        if "ROW_NUMBER()" in q or "RANK()" in q or "DENSE_RANK()" in q:
            parts["b"] = 0.15; passed.append("Uses ROW_NUMBER/RANK")
        else:
            issues.append("Use ROW_NUMBER() OVER (PARTITION BY user_id...)")
        if "WITH " in q and "AS (" in q:
            parts["c"] = 0.15; passed.append("Uses CTEs")
        else:
            issues.append("Use CTEs for pipeline")
        if q.count("SELECT") <= 3:
            parts["d"] = 0.15; passed.append("Reduced subquery count")
        else:
            issues.append(f"Still has {q.count('SELECT')} SELECTs")
        if "<= 3" in q or "= 3" in q:
            parts["e"] = 0.10; passed.append("Filters top-3 per user")
        else:
            issues.append("Filter WHERE order_rank <= 3")
        if "USER_ID" in q and "ORDER_ID" in q:
            parts["f"] = 0.10; passed.append("Required columns present")
        else:
            issues.append("Must include user_id and order_id")
        if "ORDER_ITEMS" in q:
            parts["g"] = 0.10; passed.append("Joins order_items")
        else:
            issues.append("Join order_items for order_total")

    raw = round(sum(parts.values()), 3)
    score = max(0.01, min(0.99, raw))
    return {"score": score, "issues": issues, "passed_checks": passed}


def _cost(query: str, task_id: str) -> float:
    q = query.strip().upper()
    base = {"select_star_cleanup": 1_000_000, "n_plus_one_to_join": 50_000_000,
            "window_function_rewrite": 1_638_400_000}.get(task_id, 100_000)
    m = 1.0
    if "SELECT *" in q: m *= 3.0
    subs = q.count("SELECT") - 1
    if subs > 0: m *= (4.0 ** subs)
    if "DISTINCT" in q and "GROUP BY" not in q: m *= 2.0
    if "WITH " in q and " AS (" in q: m *= 0.6
    if "OVER (" in q or "OVER(" in q: m *= 0.5
    if "LIMIT" in q: m *= 0.4
    if "JOIN" in q and "SELECT *" not in q: m *= 0.8
    return round(base * m * 0.01, 2)


class Env:
    def __init__(self, task_id="select_star_cleanup"):
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.step_num = 0
        self.max_steps = 10
        self.done = False
        self.current_query = self.task["original_query"]
        self.orig_cost = _cost(self.task["original_query"], task_id)
        self.cumulative = 0.0
        self.last_score = 0.0
        self.t0 = time.time()

    def obs(self, grade=None, reward=0.0, plan="seq_scan",
            est_cost=None, ratio=None, valid=True, err=None):
        return {
            "task_id": self.task_id,
            "difficulty": self.task["difficulty"],
            "description": self.task["description"],
            "original_query": self.task["original_query"],
            "current_query": self.current_query,
            "step_number": self.step_num,
            "max_steps": self.max_steps,
            "syntax_valid": valid,
            "execution_error": err,
            "query_plan": {"plan_type": plan, "estimated_cost": est_cost or 0},
            "estimated_cost": est_cost,
            "original_cost": self.orig_cost,
            "cost_improvement_ratio": ratio,
            "issues_found": grade["issues"] if grade else [],
            "passed_checks": grade["passed_checks"] if grade else [],
            "hints": self.task["hints"] if self.step_num <= 3 else [],
            "last_reward": reward,
            "cumulative_reward": round(self.cumulative, 4),
        }

    def step(self, query: str):
        self.step_num += 1
        q = query.strip().upper()
        if not q or not (q.startswith("SELECT") or q.startswith("WITH")) \
                or q.count("(") != q.count(")"):
            r = -0.05
            self.cumulative += r
            self.current_query = query
            done = self.step_num >= self.max_steps
            self.done = done
            return self.obs(reward=r, valid=False,
                            err="Query must start with SELECT or WITH"), r, done, {}
        grade = _grade(self.task_id, query)
        gs = grade["score"]
        ec = _cost(query, self.task_id)
        ratio = round(self.orig_cost / max(ec, 1.0), 3)
        plan = ("window_agg" if "OVER" in q else "hash_join" if "JOIN" in q
                else "seq_scan" if "SELECT *" in q else "index_scan")
        delta = gs - self.last_score
        progress = (min(delta, 0.10) if delta > 0.05 else
                    -0.03 if delta < -0.10 else 0.0)
        r = round(min(max(gs * 0.6 + (min(ratio, 10) / 10) * 0.3 + progress, 0.01), 0.99), 4)
        self.current_query = query
        self.cumulative += r
        self.last_score = gs
        done = self.step_num >= self.max_steps or gs >= 0.95
        self.done = done
        return self.obs(grade=grade, reward=r, plan=plan,
                        est_cost=ec, ratio=ratio), r, done, {"grade": grade}


_env: Optional[Env] = None


def get_env() -> Env:
    global _env
    if _env is None:
        _env = Env("select_star_cleanup")
    return _env


@app.get("/")
def root():
    return {"status": "ok", "message": "SQL Optimization OpenEnv is running.",
            "version": "2.0.0",
            "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"]}


@app.get("/health")
def health():
    return {"status": "ok", "env": "sql_opt_env", "version": "2.0.0"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": [
        {"task_id": tid, "difficulty": t["difficulty"], "description": t["description"]}
        for tid, t in TASKS.items()
    ]}


@app.post("/reset")
async def reset(request: Request):
    global _env
    task_id = "select_star_cleanup"
    try:
        raw = await request.body()
        if raw and raw.strip() not in (b"", b"{}"):
            body = json.loads(raw)
            if isinstance(body, dict) and body.get("task_id") in TASKS:
                task_id = body["task_id"]
    except Exception:
        pass
    _env = Env(task_id)
    return JSONResponse({"observation": _env.obs(), "task_id": task_id,
                         "status": "ok", "info": {"task_id": task_id}})


@app.post("/step")
async def step_route(request: Request):
    query = ""
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw)
            query = body.get("query", "")
    except Exception:
        pass
    if not query:
        return JSONResponse(status_code=400,
                            content={"error": "query required", "status": "error"})
    obs, reward, done, info = get_env().step(query)
    return JSONResponse({"observation": obs, "reward": reward,
                         "done": done, "info": info, "status": "ok"})


@app.get("/state")
def state():
    e = get_env()
    return JSONResponse({"task_id": e.task_id, "step": e.step_num,
                         "max_steps": e.max_steps, "done": e.done,
                         "cumulative_reward": e.cumulative,
                         "elapsed_seconds": round(time.time() - e.t0, 2),
                         "status": "ok"})


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == '__main__':
    main()