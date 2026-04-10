"""
server.py - SQL Query Optimization OpenEnv
SELF-CONTAINED: zero external package imports.
All env logic is inline. Works regardless of sql_opt_env package state.
"""

import os
import json
import time
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="SQL Query Optimization OpenEnv", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Inline task definitions ───────────────────────────────────────────────────

TASKS = {
    "select_star_cleanup": {
        "task_id": "select_star_cleanup",
        "difficulty": "easy",
        "description": (
            "The dashboard query uses SELECT DISTINCT * on a 1M-row users table "
            "to display only username and email. Optimize it: remove SELECT *, "
            "remove DISTINCT, add is_active filter, add LIMIT."
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
            "A correlated subquery in SELECT runs once per user row (N+1 problem). "
            "With 1M users this is catastrophic. Rewrite using JOIN + GROUP BY or CTE. "
            "Output: user_id, username, total_spent."
        ),
        "original_query": (
            "SELECT u.user_id, u.username,\n"
            "    (SELECT SUM(o.total_amount) FROM orders o\n"
            "     WHERE o.user_id = u.user_id AND o.status = 'completed') AS total_spent\n"
            "FROM users u WHERE u.is_active = TRUE;"
        ),
        "hints": [
            "Replace correlated subquery with LEFT JOIN orders GROUP BY",
            "Or use CTE: WITH totals AS (SELECT user_id, SUM(total_amount)...)",
            "Preserve status = 'completed' and is_active filters",
        ],
    },
    "window_function_rewrite": {
        "task_id": "window_function_rewrite",
        "difficulty": "hard",
        "description": (
            "Four nested correlated subqueries for ranking and running totals - O(N^2). "
            "Rewrite using ROW_NUMBER() OVER and SUM() OVER window functions with CTEs. "
            "Output: user_id, order_id, order_rank, order_total, running_total_spend. Top 3 per user."
        ),
        "original_query": (
            "SELECT u.user_id, o.order_id,\n"
            "    (SELECT COUNT(*) FROM orders o2 WHERE o2.user_id = o.user_id\n"
            "       AND o2.created_at >= o.created_at) AS order_rank,\n"
            "    (SELECT SUM(oi.unit_price * oi.quantity) FROM order_items oi\n"
            "     WHERE oi.order_id = o.order_id) AS order_total,\n"
            "    (SELECT SUM(o3.total_amount) FROM orders o3\n"
            "     WHERE o3.user_id = o.user_id AND o3.created_at <= o.created_at\n"
            "       AND (SELECT COUNT(*) FROM orders o4 WHERE o4.user_id = o.user_id\n"
            "            AND o4.created_at >= o3.created_at) <= 3) AS running_total_spend\n"
            "FROM users u JOIN orders o ON u.user_id = o.user_id\n"
            "WHERE u.is_active = TRUE\n"
            "  AND (SELECT COUNT(*) FROM orders o5 WHERE o5.user_id = o.user_id\n"
            "       AND o5.created_at >= o.created_at) <= 3;"
        ),
        "hints": [
            "Use ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC)",
            "Use SUM() OVER (PARTITION BY user_id ORDER BY order_rank) for running total",
            "Pre-aggregate order_items in a CTE first",
            "Pipeline: order_totals CTE -> ranked_orders CTE -> filter rank<=3 -> window SUM",
        ],
    },
}


def _grade_query(task_id: str, query: str) -> Dict[str, Any]:
    """Deterministic grader for each task."""
    q = query.strip().upper()
    issues, passed, score_parts = [], [], {}

    if not (q.startswith("SELECT") or q.startswith("WITH")):
        return {"score": 0.0, "issues": ["Not a SELECT or WITH statement"], "passed_checks": []}

    if task_id == "select_star_cleanup":
        if "SELECT *" not in q:
            score_parts["no_star"] = 0.25; passed.append("Removed SELECT *")
        else:
            issues.append("Still using SELECT * — fetches unused TEXT columns")
        if "DISTINCT" not in q:
            score_parts["no_distinct"] = 0.15; passed.append("Removed DISTINCT")
        else:
            issues.append("DISTINCT is unnecessary on a PRIMARY KEY table")
        if "IS_ACTIVE" in q:
            score_parts["active_filter"] = 0.20; passed.append("Filters on is_active (indexed)")
        else:
            issues.append("Add WHERE is_active = TRUE — it's indexed and reduces rows")
        if "COUNTRY" in q and ("'US'" in q or '"US"' in q):
            score_parts["country_filter"] = 0.10; passed.append("Preserves country filter")
            if "USERNAME" in q and "EMAIL" in q:
                score_parts["columns"] = 0.20; passed.append("Selects only username, email")
            else:
                issues.append("Select only username and email")
        else:
            issues.append("Preserve WHERE country = 'US'")
        if "LIMIT" in q:
            score_parts["limit"] = 0.10; passed.append("Added LIMIT")
        else:
            issues.append("Add LIMIT for pagination")

    elif task_id == "n_plus_one_to_join":
        select_count = q.count("SELECT")
        has_corr = select_count >= 2 and "GROUP BY" not in q
        if select_count == 1:
            score_parts["no_corr"] = 0.30; passed.append("Eliminated correlated subquery")
        elif "WITH " in q and select_count == 2 and not has_corr:
            score_parts["no_corr"] = 0.20; passed.append("CTE eliminates N+1")
        else:
            issues.append("Still has correlated subquery in SELECT — N+1 persists")
        if "JOIN" in q or ("WITH " in q and "AS (" in q):
            score_parts["join_cte"] = 0.20; passed.append("Uses JOIN or CTE")
        else:
            issues.append("Use JOIN or CTE for set-based aggregation")
        if "USER_ID" in q and "USERNAME" in q:
            score_parts["output"] = 0.15; passed.append("Returns user_id, username")
        else:
            issues.append("Must return user_id and username")
        if "'COMPLETED'" in q or '"COMPLETED"' in q:
            score_parts["status"] = 0.15; passed.append("Preserves status filter")
        else:
            issues.append("Missing status = 'completed' filter")
        if "IS_ACTIVE" in q:
            score_parts["active"] = 0.10; passed.append("Preserves is_active filter")
        else:
            issues.append("Missing is_active filter")
        if "SUM(" in q:
            score_parts["sum"] = 0.10; passed.append("Uses SUM aggregation")
        else:
            issues.append("Missing SUM for total_spent")

    elif task_id == "window_function_rewrite":
        if "OVER (" in q or "OVER(" in q:
            score_parts["window"] = 0.25; passed.append("Uses window functions")
        else:
            issues.append("Must use OVER clause for window functions")
        if "ROW_NUMBER()" in q or "RANK()" in q or "DENSE_RANK()" in q:
            score_parts["row_num"] = 0.15; passed.append("Uses ROW_NUMBER/RANK")
        else:
            issues.append("Use ROW_NUMBER() OVER (PARTITION BY user_id...)")
        if "WITH " in q and "AS (" in q:
            score_parts["cte"] = 0.15; passed.append("Uses CTEs")
        else:
            issues.append("Use CTEs for pipeline clarity")
        if q.count("SELECT") <= 3:
            score_parts["reduced"] = 0.15; passed.append("Reduced subquery count")
        else:
            issues.append(f"Still has {q.count('SELECT')} SELECTs — original had 5")
        if "<= 3" in q or "= 3" in q:
            score_parts["top3"] = 0.10; passed.append("Filters top-3 per user")
        else:
            issues.append("Filter WHERE order_rank <= 3")
        if "USER_ID" in q and "ORDER_ID" in q:
            score_parts["cols"] = 0.10; passed.append("Required columns present")
        else:
            issues.append("Must include user_id and order_id")
        if "ORDER_ITEMS" in q:
            score_parts["items"] = 0.10; passed.append("Joins order_items")
        else:
            issues.append("Join order_items to compute order_total")

    return {
        "score": round(min(1.0, sum(score_parts.values())), 3),
        "issues": issues,
        "passed_checks": passed,
    }


def _simulate_cost(query: str, task_id: str) -> float:
    """Heuristic cost model."""
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


# ── In-memory environment state ───────────────────────────────────────────────

class EnvState:
    def __init__(self, task_id: str = "select_star_cleanup"):
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.step = 0
        self.max_steps = 10
        self.done = False
        self.current_query = self.task["original_query"]
        self.original_cost = _simulate_cost(self.task["original_query"], task_id)
        self.cumulative_reward = 0.0
        self.last_grade_score = 0.0
        self.start_time = time.time()

    def observation(self, grade=None, reward=0.0, plan_type="seq_scan",
                    estimated_cost=None, cost_ratio=None, syntax_valid=True, error=None):
        return {
            "task_id": self.task_id,
            "difficulty": self.task["difficulty"],
            "description": self.task["description"],
            "original_query": self.task["original_query"],
            "current_query": self.current_query,
            "step_number": self.step,
            "max_steps": self.max_steps,
            "syntax_valid": syntax_valid,
            "execution_error": error,
            "estimated_cost": estimated_cost,
            "original_cost": self.original_cost,
            "cost_improvement_ratio": cost_ratio,
            "query_plan": {"plan_type": plan_type, "estimated_cost": estimated_cost or 0},
            "issues_found": grade["issues"] if grade else [],
            "hints": self.task["hints"] if self.step <= 3 else [],
            "last_reward": reward,
            "cumulative_reward": round(self.cumulative_reward, 4),
        }

    def step_env(self, query: str):
        self.step += 1
        q = query.strip().upper()

        # Syntax check
        if not q or not (q.startswith("SELECT") or q.startswith("WITH")):
            reward = -0.05
            self.cumulative_reward += reward
            self.current_query = query
            done = self.step >= self.max_steps
            self.done = done
            obs = self.observation(reward=reward, syntax_valid=False,
                                   error="Query must start with SELECT or WITH")
            return obs, reward, done, {"error": "syntax"}

        if q.count("(") != q.count(")"):
            reward = -0.05
            self.cumulative_reward += reward
            self.current_query = query
            done = self.step >= self.max_steps
            self.done = done
            obs = self.observation(reward=reward, syntax_valid=False,
                                   error="Unbalanced parentheses")
            return obs, reward, done, {"error": "syntax"}

        grade = _grade_query(self.task_id, query)
        grade_score = grade["score"]
        estimated_cost = _simulate_cost(query, self.task_id)
        cost_ratio = round(self.original_cost / max(estimated_cost, 1.0), 3)

        # Plan type
        if "OVER" in q: plan_type = "window_agg"
        elif "JOIN" in q: plan_type = "hash_join"
        elif "SELECT *" in q: plan_type = "seq_scan"
        else: plan_type = "index_scan"

        # Reward
        grade_comp = grade_score * 0.6
        cost_comp = (min(cost_ratio, 10.0) / 10.0) * 0.3
        delta = grade_score - self.last_grade_score
        progress = min(delta, 0.10) if delta > 0.05 else (-0.03 if delta < -0.10 else 0.0)
        reward = round(min(max(grade_comp + cost_comp + progress, 0.0), 1.0), 4)

        self.current_query = query
        self.cumulative_reward += reward
        self.last_grade_score = grade_score
        done = self.step >= self.max_steps or grade_score >= 0.95
        self.done = done

        obs = self.observation(grade=grade, reward=reward, plan_type=plan_type,
                               estimated_cost=estimated_cost, cost_ratio=cost_ratio)
        return obs, reward, done, {"grade": grade, "cost_improvement_ratio": cost_ratio}


_state: Optional[EnvState] = None

def get_state() -> EnvState:
    global _state
    if _state is None:
        _state = EnvState("select_star_cleanup")
    return _state


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SQL Optimization OpenEnv is running.",
        "version": "2.0.0",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "sql_opt_env", "version": "2.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"task_id": tid, "difficulty": t["difficulty"], "description": t["description"]}
            for tid, t in TASKS.items()
        ]
    }


@app.post("/reset")
async def reset(request: Request):
    """
    Handles POST /reset with any body — empty {}, null, or {"task_id":"..."}.
    Always returns HTTP 200 with task_id in response.
    """
    global _state
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

    _state = EnvState(task_id)
    obs = _state.observation()
    return JSONResponse(content={
        "observation": obs,
        "task_id": task_id,
        "status": "ok",
        "info": {"task_id": task_id},
    })


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
        return JSONResponse(status_code=400,
                            content={"error": "query field is required", "status": "error"})

    obs, reward, done, info = get_state().step_env(query)
    return JSONResponse(content={
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
        "status": "ok",
    })


@app.get("/state")
def state():
    s = get_state()
    return JSONResponse(content={
        "task_id": s.task_id,
        "step": s.step,
        "max_steps": s.max_steps,
        "done": s.done,
        "cumulative_reward": s.cumulative_reward,
        "current_query": s.current_query,
        "elapsed_seconds": round(time.time() - s.start_time, 2),
        "status": "ok",
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    print(f"[server] Starting on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
