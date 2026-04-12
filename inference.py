"""
PERSON B — inference.py
=======================
Baseline inference script for the SQL Query Optimization OpenEnv.
"""

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sql_opt_env import SQLOptEnv, SQLOptAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK         = "sql_opt_env"
MAX_STEPS         = 6
TEMPERATURE       = 0.2
MAX_TOKENS        = 800
SUCCESS_THRESHOLD = 0.5

TASK_IDS = [
    "select_star_cleanup",
    "n_plus_one_to_join",
    "window_function_rewrite",
]


def _clamp(value: float) -> float:
    """Clamp to strictly (0, 1) — never 0.0 or 1.0 exactly."""
    return round(min(max(float(value), 0.01), 0.99), 4)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    if len(action_clean) > 300:
        action_clean = action_clean[:297] + "..."
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent("""
You are a senior database engineer and SQL optimization expert.
Your job is to rewrite slow, inefficient SQL queries into fast, production-grade ones.

You will receive:
  - A task description explaining the business requirement
  - The database schema (tables, columns, row counts, which columns are indexed)
  - The original slow query to improve
  - Your current best query
  - Feedback: issues found, cost improvement so far, hints

Your goal is to produce a query that:
  1. Returns EXACTLY the same result set as the original (semantics preserved)
  2. Uses efficient access patterns (indexed columns, set-based operations)
  3. Avoids anti-patterns (SELECT *, correlated subqueries, unnecessary DISTINCT)
  4. Uses modern SQL constructs where appropriate (CTEs, window functions)

OUTPUT RULES — critical:
  - Return ONLY the SQL query. Nothing else.
  - Do NOT wrap in markdown backticks or code blocks.
  - Do NOT add any explanation, preamble, or commentary.
  - The query MUST start with SELECT or WITH
  - Preserve all WHERE filters from the original (changing filters changes results)
""").strip()


def _format_schemas(obs_dict: dict) -> str:
    schemas = obs_dict.get("schemas", [])
    lines = []
    for s in schemas:
        lines.append(f"\nTable: {s['name']} ({s['row_count']:,} rows) — {s['description']}")
        lines.append("Columns:")
        for col in s["columns"]:
            idx_note = f" [{col['indexed']}]" if col.get("indexed", "NO") != "NO" else ""
            lines.append(f"  {col['name']}  {col['type']}{idx_note}")
    return "\n".join(lines)


def build_user_prompt(obs_dict: dict, step: int, last_reward: float, history: List[str]) -> str:
    issues_block = "\n".join(f"  ⚠  {i}" for i in obs_dict.get("issues_found", []))
    hints_block  = "\n".join(f"  💡 {h}" for h in obs_dict.get("hints", []))
    hist_block   = "\n".join(f"  {h}" for h in history[-3:]) if history else "  (none yet)"
    cr           = obs_dict.get("cost_improvement_ratio")
    cost_str     = f"{cr:.2f}× improvement over baseline" if cr else "not yet measured"

    return textwrap.dedent(f"""
    ═══════════════════════════════════════════════════════
    TASK  [{obs_dict.get('difficulty', '').upper()}]: {obs_dict.get('task_id', '')}
    ═══════════════════════════════════════════════════════
    {obs_dict.get('description', '')}

    DATABASE SCHEMA:
    {_format_schemas(obs_dict)}

    ───────────────────────────────────────────────────────
    ORIGINAL (SLOW) QUERY:
    {obs_dict.get('original_query', '')}

    ───────────────────────────────────────────────────────
    YOUR CURRENT QUERY (step {step}):
    {obs_dict.get('current_query', '')}

    ───────────────────────────────────────────────────────
    FEEDBACK:
    Step reward:        {last_reward:.2f} / 1.00
    Cost improvement:   {cost_str}

    Issues still present:
    {issues_block if issues_block else "  ✓ No issues detected — you may refine further"}

    Optimization hints:
    {hints_block if hints_block else "  (no hints at this stage)"}

    Recent history:
    {hist_block}
    ───────────────────────────────────────────────────────
    Write your improved SQL query now:
    """).strip()


def get_model_query(client, obs_dict: dict, step: int, last_reward: float, history: List[str]) -> str:
    prompt = build_user_prompt(obs_dict, step, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        if text and (text.upper().startswith("SELECT") or text.upper().startswith("WITH")):
            return text

        print(f"[DEBUG] Model returned non-SQL response, falling back to current query", flush=True)
        return obs_dict.get("current_query") or obs_dict.get("original_query", "SELECT 1")

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return obs_dict.get("current_query") or obs_dict.get("original_query", "SELECT 1")


def run_task(client: OpenAI, task_id: str) -> dict:
    env = SQLOptEnv(task_id=task_id)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0

    # Safe defaults — must exist before try block so finally always works
    score = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()
        obs_dict = observation.model_dump()
        last_reward = 0.01

        for step_num in range(1, MAX_STEPS + 1):
            query = get_model_query(client, obs_dict, step_num, last_reward, history)
            observation, reward, done, info = env.step(SQLOptAction(query=query))
            obs = observation
            error = info.get("execution_error")

            # Clamp every individual reward strictly within (0, 1)
            reward = _clamp(reward)

            rewards.append(reward)
            steps_taken = step_num
            last_reward = reward
            obs_dict = obs.model_dump()

            log_step(step=step_num, action=query, reward=reward, done=done, error=error)

            issues_summary = obs.issues_found[:2] if obs.issues_found else ["no issues"]
            history.append(f"Step {step_num}: reward={reward:.2f} issues={issues_summary}")

            if done:
                break

        raw_score = max(rewards) if rewards else 0.01
        score = _clamp(raw_score)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required for inference.py")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_results = []
    for task_id in TASK_IDS:
        print(f"\n{'═'*65}", flush=True)
        print(f"  Running: {task_id}", flush=True)
        print(f"{'═'*65}", flush=True)
        result = run_task(client, task_id)
        all_results.append(result)

    print(f"\n{'═'*65}", flush=True)
    print("  BASELINE SUMMARY", flush=True)
    print(f"{'─'*65}", flush=True)
    print(f"  {'Task':<35} {'Score':>7}  {'Steps':>5}  {'Status'}", flush=True)
    print(f"{'─'*65}", flush=True)
    for r in all_results:
        status = "PASS ✓" if r["success"] else "FAIL ✗"
        print(f"  {r['task_id']:<35} {r['score']:>7.3f}  {r['steps']:>5}  {status}", flush=True)
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"{'─'*65}", flush=True)
    print(f"  {'Average':<35} {avg:>7.3f}", flush=True)
    print(f"{'═'*65}\n", flush=True)


if __name__ == "__main__":
    main()