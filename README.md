---
title: SQL Query Optimization OpenEnv
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🗄️ SQL Query Optimization Environment

**Team Marvel** | Meta PyTorch Hackathon × Scaler School of Technology

An RL environment where AI agents learn to rewrite slow, production-breaking SQL queries into fast, efficient ones — across 3 real-world difficulty levels.

---

## 🎯 What This Environment Does

SQL performance bugs are everywhere in production. A correlated subquery on a 50M-row table can take minutes. An unnecessary `SELECT *` wastes memory. This environment trains agents to identify and fix these patterns automatically.

The agent receives a broken/inefficient SQL query, submits an optimized version, and receives a graded reward based on how much it improved.

---

## 🏗️ Architecture

```
sql-open-env/
├── server/
│   └── app.py          # FastAPI server — reset, step, state endpoints
├── sql_opt_env/        # Environment core logic
├── inference.py        # LLM agent — runs optimization loop
├── openenv.yaml        # Environment manifest
├── pyproject.toml      # Package config with server entry point
└── Dockerfile          # Container for HF Spaces deployment
```

---

## 📋 Tasks

| Task | Difficulty | Problem | Target Improvement |
|------|-----------|---------|-------------------|
| `select_star_cleanup` | Easy | `SELECT DISTINCT *` on 1M-row table | Remove `*`, DISTINCT, add indexed filter + LIMIT |
| `n_plus_one_to_join` | Medium | Correlated subquery runs N+1 times | Rewrite with JOIN + GROUP BY or CTE |
| `window_function_rewrite` | Hard | 4 nested correlated subqueries O(N²) | Rewrite with ROW_NUMBER OVER, SUM OVER, CTEs |

---

## 🔄 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode, returns initial observation |
| `/step` | POST | Submit an optimized query, returns reward + feedback |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Health check |

---

## 📊 Observation Space

```json
{
  "task_id": "select_star_cleanup",
  "difficulty": "easy",
  "original_query": "SELECT DISTINCT * FROM users ...",
  "current_query": "SELECT username, email FROM users ...",
  "step_number": 2,
  "max_steps": 10,
  "syntax_valid": true,
  "issues_found": ["Add LIMIT for pagination"],
  "passed_checks": ["Removed SELECT *", "Removed DISTINCT"],
  "hints": ["Add LIMIT 100"],
  "last_reward": 0.72,
  "cumulative_reward": 1.15
}
```

---

## ⚡ Action Space

```json
{
  "query": "SELECT username, email FROM users WHERE country = 'US' AND is_active = TRUE ORDER BY created_at DESC LIMIT 100",
  "explanation": "Removed SELECT *, DISTINCT, added is_active filter and LIMIT"
}
```

---

## 🏆 Reward Function

Rewards are shaped and dense — the agent gets feedback every step, not just at the end.

| Check | Points | Description |
|-------|--------|-------------|
| Remove SELECT * | +0.25 | Specify only needed columns |
| Remove DISTINCT | +0.15 | Unnecessary on PRIMARY KEY |
| Add is_active filter | +0.20 | Indexed column, massive speedup |
| Preserve country filter | +0.10 | Don't break existing logic |
| Select username, email | +0.20 | Minimal projection |
| Add LIMIT | +0.10 | Essential for pagination |

**Score range:** strictly (0.01, 0.99) — shaped for RL training signal at every step.

---

## 🚀 Quick Start

### Connect via Python

```python
from sql_opt_env import SQLOptEnv, SQLOptAction

env = SQLOptEnv(task_id="select_star_cleanup")
obs = env.reset()
print(obs.original_query)

obs, reward, done, info = env.step(SQLOptAction(
    query="SELECT username, email FROM users WHERE country = 'US' AND is_active = TRUE LIMIT 100",
    explanation="Removed SELECT *, DISTINCT, added filter and LIMIT"
))
print(f"Reward: {reward:.3f}")
print(f"Issues: {obs.issues_found}")
```

### Run the LLM Agent

```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

### Connect via HTTP

```bash
# Reset environment
curl -X POST https://winniem30-sql-open-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "select_star_cleanup"}'

# Submit optimized query
curl -X POST https://winniem30-sql-open-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT username, email FROM users WHERE country = '\''US'\'' AND is_active = TRUE LIMIT 100"}'
```

---

## 🧠 Why This Environment Matters

SQL optimization is a real, high-value problem:

- **Production impact** — a bad query on a 50M-row table can take 10+ minutes and crash services
- **Generalizable** — the 3 tasks cover the most common real-world SQL anti-patterns
- **Rich signal** — dense rewards at every step make it ideal for RL training
- **Realistic** — based on actual patterns seen in production codebases

---

## 📈 Baseline Performance

| Agent | Easy | Medium | Hard | Average |
|-------|------|--------|------|---------|
| Random queries | ~0.10 | ~0.05 | ~0.03 | ~0.06 |
| LLM 1-shot | ~0.65 | ~0.45 | ~0.30 | ~0.47 |
| LLM 8-step iterative | ~0.85 | ~0.70 | ~0.55 | ~0.70 |
| Oracle (perfect) | 0.99 | 0.99 | 0.99 | 0.99 |

---

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | required | Hugging Face API token |
| `API_BASE_URL` | HF Router | LLM API endpoint |
| `MODEL_NAME` | Qwen 72B | Model identifier |
| `PORT` | 7860 | Server port |

---

*Built for the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, 2026*