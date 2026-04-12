"""
inference.py — SQL Query Optimization OpenEnv
Team Marvel | Meta PyTorch Hackathon x Scaler School of Technology

Agent: LLM-powered SQL optimizer using iterative refinement.
Logs: Structured [START], [STEP], [END] format as required by evaluator.
"""

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sql_opt_env import SQLOptEnv, SQLOptAction

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK         = "sql_opt_env"
MAX_STEPS         = 8
TEMPERATURE       = 0.15
MAX_TOKENS        = 1000
SUCCESS_THRESHOLD = 0.50

TASK_IDS = [
    "select_star_cleanup",
    "n_plus_one_to_join",
    "window_function_rewrite",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _clamp(value: float) -> float:
    """Strictly (0, 1) — never 0.0 or 1.0 exactly."""
    return round(min(max(float(value), 0.01), 0.99), 4)


# ── MANDATORY LOG FUNCTIONS ───────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    if len(action_clean) > 300:
        action_clean = action_clean[:297] + "..."
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── PROMPTS ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior database engineer and SQL optimization expert with 15+ years experience.
Your job is to rewrite slow, inefficient SQL queries into fast, production-grade ones.

OPTIMIZATION PRINCIPLES:
  1. Replace SELECT * with specific column names
  2. Remove unnecessary DISTINCT (especially on PRIMARY KEY tables)
  3. Replace correlated subqueries with JOINs or CTEs
  4. Use window functions (ROW_NUMBER, SUM OVER) for ranking/running totals
  5. Add selective WHERE filters on indexed columns (e.g. is_active = TRUE)
  6. Add LIMIT for pagination queries
  7. Use CTEs (WITH clause) for complex multi-step queries

OUTPUT RULES — strictly follow these:
  - Return ONLY the SQL query. Nothing else.
  - Do NOT wrap in markdown backticks or code blocks.
  - Do NOT add any explanation, preamble, or commentary.
  - The query MUST start with SELECT or WITH
  - Preserve all original WHERE filters
""").strip()


def _format_issues(issues: list) -> str:
    if not issues:
        return "  No issues — query looks good!"
    return "\n".join(f"  - {i}" for i in issues)


def _format_hints(hints: list) -> str:
    if not hints:
        return "  (no hints available)"
    return "\n".join(f"  - {h}" for h in hints)


def _format_history(history: list) -> str:
    if not history:
        return "  (first attempt)"
    return "\n".join(f"  {h}" for h in history[-3:])


def build_user_prompt(obs_dict: dict, step: int, last_reward: float, history: List[str]) -> str:
    difficulty = obs_dict.get("difficulty", "").upper()
    task_id    = obs_dict.get("task_id", "")
    desc       = obs_dict.get("description", "")
    orig_q     = obs_dict.get("original_query", "")
    curr_q     = obs_dict.get("current_query", "")
    issues     = obs_dict.get("issues_found", [])
    hints      = obs_dict.get("hints", [])
    passed     = obs_dict.get("passed_checks", [])

    passed_block = "\n".join(f"  - {p}" for p in passed) if passed else "  (none yet)"

    if last_reward < 0.20:
        urgency = "Your current query needs major improvement. Focus on the issues below."
    elif last_reward < 0.50:
        urgency = "Good progress! Address the remaining issues to push score higher."
    else:
        urgency = "Almost there! Fix the remaining issues for a perfect score."

    return textwrap.dedent(f"""
    TASK [{difficulty}]: {task_id}
    {desc}

    ORIGINAL QUERY:
    {orig_q}

    YOUR CURRENT QUERY (step {step}):
    {curr_q}

    PROGRESS: reward={last_reward:.2f} | {urgency}

    PASSED CHECKS:
    {passed_block}

    ISSUES TO FIX:
    {_format_issues(issues)}

    HINTS:
    {_format_hints(hints)}

    RECENT HISTORY:
    {_format_history(history)}

    Write your improved SQL query now (SQL only, no explanation):
    """).strip()


# ── LLM CALL ─────────────────────────────────────────────────────────────────

def get_model_query(
    client: OpenAI,
    obs_dict: dict,
    step: int,
    last_reward: float,
    history: List[str],
) -> str:
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

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        if text and (text.upper().startswith("SELECT") or text.upper().startswith("WITH")):
            return text

        print(f"[DEBUG] Non-SQL response at step {step}, using current query", flush=True)
        return obs_dict.get("current_query") or obs_dict.get("original_query", "SELECT 1")

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return obs_dict.get("current_query") or obs_dict.get("original_query", "SELECT 1")


# ── TASK RUNNER ───────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> dict:
    env = SQLOptEnv(task_id=task_id)
    history: List[str]   = []
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.01
    success     = False
    best_query  = ""
    best_reward = 0.01

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()
        obs_dict    = observation.model_dump()
        last_reward = 0.01

        for step_num in range(1, MAX_STEPS + 1):
            query = get_model_query(client, obs_dict, step_num, last_reward, history)
            observation, reward, done, info = env.step(SQLOptAction(query=query))

            error       = info.get("execution_error")
            reward      = _clamp(reward)
            last_reward = reward

            rewards.append(reward)
            steps_taken = step_num
            obs_dict    = observation.model_dump()

            if reward > best_reward:
                best_reward = reward
                best_query  = query

            log_step(step=step_num, action=query, reward=reward, done=done, error=error)

            issues_summary = obs_dict.get("issues_found", [])[:2] or ["no issues"]
            passed_summary = obs_dict.get("passed_checks", [])[:2] or []
            history.append(
                f"Step {step_num}: reward={reward:.2f} "
                f"passed={passed_summary} issues={issues_summary}"
            )

            if done:
                break

        raw_score = max(rewards) if rewards else 0.01
        score     = _clamp(raw_score)
        success   = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":    task_id,
        "score":      score,
        "success":    success,
        "steps":      steps_taken,
        "rewards":    rewards,
        "best_query": best_query,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable is required.\n"
            "Set it with: export HF_TOKEN=your_token_here"
        )

    client      = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    all_results = []

    print(f"\n{'='*60}", flush=True)
    print(f"  SQL Query Optimization Agent", flush=True)
    print(f"  Model : {MODEL_NAME}", flush=True)
    print(f"  Tasks : {len(TASK_IDS)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for task_id in TASK_IDS:
        print(f"\n{'─'*60}", flush=True)
        print(f"  Task: {task_id}", flush=True)
        print(f"{'─'*60}", flush=True)
        result = run_task(client, task_id)
        all_results.append(result)

    print(f"\n{'='*60}", flush=True)
    print("  FINAL SUMMARY", flush=True)
    print(f"{'─'*60}", flush=True)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task_id']:<35} score={r['score']:.3f}  {status}", flush=True)
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"{'─'*60}", flush=True)
    print(f"  Average score : {avg:.3f}", flush=True)
    passed = sum(1 for r in all_results if r["success"])
    print(f"  Tasks passed  : {passed}/{len(all_results)}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()