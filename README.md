# SQL Query Optimization OpenEnv

An RL environment where AI agents learn to rewrite slow SQL queries into efficient, production-grade ones. This simulates a real database engineering task: given a query with anti-patterns (SELECT *, correlated subqueries, missing index usage), the agent must iteratively optimize it using modern SQL constructs (CTEs, window functions, proper JOINs).

## Project Overview

This OpenEnv environment challenges AI agents to optimize SQL queries by iteratively improving them based on feedback about syntax validity, semantic correctness, and estimated execution cost. The environment provides realistic database schemas with row counts and indexing information, making it suitable for training agents on practical database optimization tasks.

## Motivation

SQL query optimization is a critical skill in data engineering and backend development. Poorly written queries can cause massive performance degradation in production systems, leading to slow response times, high resource consumption, and poor user experience. This environment teaches agents to:

- Identify common SQL anti-patterns (SELECT *, correlated subqueries, unnecessary operations)
- Apply optimization techniques (column selection, JOIN rewriting, window functions)
- Balance correctness with performance improvements
- Use database metadata (indexes, row counts) to make informed decisions

## Action Space

Actions are SQL queries submitted as strings:

```python
{
  "query": "SELECT username, email FROM users WHERE is_active = TRUE",
  "explanation": "Optional explanation of the optimization"
}
```

Queries must start with `SELECT` or `WITH` and return semantically equivalent results to the original query.

## Observation Space

Observations provide complete context for optimization:

```python
{
  "task_id": "select_star_cleanup",
  "difficulty": "easy",
  "description": "Replace SELECT * with specific columns and preserve is_active filtering",
  "schemas": [...],  # Table schemas with columns, types, indexes, row counts
  "original_query": "SELECT * FROM users",
  "current_query": "SELECT * FROM users",
  "step_number": 0,
  "max_steps": 10,
  "syntax_valid": true,
  "execution_error": null,
  "estimated_cost": 20000.0,
  "original_cost": 20000.0,
  "cost_improvement_ratio": null,
  "issues_found": [],
  "hints": ["Avoid SELECT * and list only the needed columns."],
  "last_reward": 0.0,
  "cumulative_reward": 0.0
}
```

## Tasks

### Easy: SELECT * Cleanup (`select_star_cleanup`)
**Objective**: Replace `SELECT *` with specific columns and preserve business constraints.

**Original Query**: `SELECT * FROM users`

**Optimal Solution**: `SELECT id, username, email FROM users WHERE is_active = TRUE`

**Key Optimizations**:
- Remove `SELECT *` to avoid unnecessary data transfer
- Add `is_active = TRUE` filter to reduce scanned rows
- Select only needed columns

### Medium: N+1 to JOIN (`n_plus_one_to_join`)
**Objective**: Convert correlated subquery causing N+1 execution into efficient JOIN + GROUP BY.

**Original Query**: `SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE country = 'US')`

**Optimal Solution**:
```sql
SELECT o.id, o.user_id, o.total
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.country = 'US'
```

**Key Optimizations**:
- Replace `IN (SELECT ...)` with JOIN
- Use indexed columns for JOIN conditions
- Aggregate results efficiently

### Hard: Window Function Rewrite (`window_function_rewrite`)
**Objective**: Replace nested correlated subqueries with window functions for O(N²) to O(N) complexity.

**Original Query**: `SELECT order_id, SUM(quantity) FROM order_items WHERE order_id IN (SELECT id FROM orders WHERE status = 'completed') GROUP BY order_id`

**Optimal Solution**:
```sql
WITH completed_orders AS (
  SELECT id FROM orders WHERE status = 'completed'
)
SELECT oi.order_id, SUM(oi.quantity) OVER (PARTITION BY oi.order_id) as total_quantity
FROM order_items oi
JOIN completed_orders co ON oi.order_id = co.id
```

**Key Optimizations**:
- Use window functions instead of correlated subqueries
- Leverage CTEs for complex filtering
- Partition by appropriate keys for efficient aggregation

## Reward Function

Rewards are shaped to provide incremental feedback:

- **Grade Component (60%)**: Semantic correctness and optimization quality score
- **Cost Component (30%)**: Execution plan cost improvement
- **Progress Bonus (10%)**: Step-over-step improvement delta

**Range**: [-0.05, 1.0]
- Negative rewards for syntax errors or destructive actions
- Zero or low rewards for no improvement
- High rewards for significant optimizations

## Setup Instructions

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd sql-opt-env
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Run tests:
```bash
python -m pytest tests/
```

### API Usage

Start the server:
```bash
python server.py
```

The API provides three endpoints:

#### GET /health
Check server status.

#### GET /tasks
List all available tasks with metadata.

#### POST /reset
Reset the environment for a new episode.
```json
{
  "task_id": "select_star_cleanup"
}
```

#### POST /step
Submit an action and get the next observation.
```json
{
  "query": "SELECT id, username FROM users WHERE is_active = TRUE",
  "explanation": "Removed SELECT * and added is_active filter"
}
```

#### GET /state
Get the current query state.

## Docker Usage

Build the container:
```bash
docker build -t sql-opt-env .
```

Run the container:
```bash
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN sql-opt-env
```

The environment will be available at `http://localhost:7860`.

## Baseline Scores

Current baseline using Qwen/Qwen2.5-72B-Instruct:

- **select_star_cleanup**: ~0.75
- **n_plus_one_to_join**: ~0.65
- **window_function_rewrite**: ~0.45
- **Average**: ~0.62

To run the baseline:
```bash
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Deployment

This environment is designed for deployment on Hugging Face Spaces:

- **Space SDK**: docker
- **Suggested Hardware**: cpu-basic
- **Port**: 7860
- **Tags**: openenv, sql, rl-environment

## Validation

Run the pre-submission validator:
```bash
bash scripts/validate-submission.sh
```

All checks must pass before submission to the OpenEnv Hackathon.