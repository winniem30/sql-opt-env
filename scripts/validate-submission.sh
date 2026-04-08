#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PERSON B — scripts/validate-submission.sh
# Pre-submission validation: 11 checks must all pass before submitting
# Usage:  bash scripts/validate-submission.sh [repo_dir]
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

REPO_DIR="${1:-.}"
if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  echo "Error: directory not found: ${1:-.}"
  exit 1
fi

# Color support
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  BOLD='\033[1m';   NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

PASS=0; FAIL=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { printf "  ${GREEN}PASS${NC}  %s\n" "$1"; PASS=$((PASS+1)); }
fail() { printf "  ${RED}FAIL${NC}  %s\n" "$1"; FAIL=$((FAIL+1)); }
hint() { printf "       ${YELLOW}hint:${NC} %s\n" "$1"; }

check() {
  local desc="$1"; local cmd="$2"; local hint_msg="${3:-}"
  if eval "$cmd" >/dev/null 2>&1; then
    pass "$desc"
  else
    fail "$desc"
    [ -n "$hint_msg" ] && hint "$hint_msg"
  fi
}

printf "\n${BOLD}══════════════════════════════════════════════════${NC}\n"
printf "${BOLD}  SQL Opt Env — Pre-Submission Validator${NC}\n"
printf "${BOLD}  Repo: %s${NC}\n" "$REPO_DIR"
printf "${BOLD}══════════════════════════════════════════════════${NC}\n\n"

# ── File existence checks ─────────────────────────────────────────────────────
check "openenv.yaml exists"          "[ -f '$REPO_DIR/openenv.yaml' ]"           "Create openenv.yaml with metadata"
check "Dockerfile exists"            "[ -f '$REPO_DIR/Dockerfile' ]"             "Create Dockerfile (see Person B deliverable)"
check "inference.py exists"          "[ -f '$REPO_DIR/inference.py' ]"           "inference.py must be in repo root"
check "requirements.txt exists"      "[ -f '$REPO_DIR/requirements.txt' ]"       "Create requirements.txt"
check "server.py exists"             "[ -f '$REPO_DIR/server.py' ]"              "Create server.py with /reset /step /state endpoints"
check "sql_opt_env package exists"   "[ -d '$REPO_DIR/sql_opt_env' ]"            "Create sql_opt_env/ directory with __init__.py"
check "tests directory exists"       "[ -d '$REPO_DIR/tests' ]"                  "Create tests/ directory"

# ── Python import checks ──────────────────────────────────────────────────────
check "Python package imports OK" \
  "cd '$REPO_DIR' && python -c 'from sql_opt_env import SQLOptEnv, SQLOptAction'" \
  "Ensure sql_opt_env/__init__.py exports SQLOptEnv and SQLOptAction"

check "All 3 tasks accessible" \
  "cd '$REPO_DIR' && python -c \"from sql_opt_env.tasks import TASKS; assert len(TASKS)==3, f'Expected 3 tasks, got {len(TASKS)}'\"" \
  "TASKS dict must contain select_star_cleanup, n_plus_one_to_join, window_function_rewrite"

# ── Environment API checks ────────────────────────────────────────────────────
check "reset() returns valid observation" \
  "cd '$REPO_DIR' && python -c \"
from sql_opt_env import SQLOptEnv
env = SQLOptEnv()
obs = env.reset()
assert obs.step_number == 0
assert obs.task_id in ['select_star_cleanup', 'n_plus_one_to_join', 'window_function_rewrite']
\"" \
  "env.reset() must return SQLOptObservation with step_number==0"

check "step() returns reward in [0, 1] for all 3 tasks" \
  "cd '$REPO_DIR' && python -c \"
from sql_opt_env import SQLOptEnv, SQLOptAction
for tid in ['select_star_cleanup', 'n_plus_one_to_join', 'window_function_rewrite']:
    env = SQLOptEnv(tid)
    env.reset()
    obs, reward, done, info = env.step(SQLOptAction(query=\\\"SELECT username, email FROM users WHERE country='US' AND is_active=TRUE LIMIT 100\\\"))
    assert 0.0 <= reward <= 1.0, f'{tid}: reward {reward} out of [0,1]'
\"" \
  "step() must return (observation, reward, done, info) with reward in [0.0, 1.0]"

check "graders are deterministic" \
  "cd '$REPO_DIR' && python -c \"
from sql_opt_env.tasks import TASKS
q = \\\"SELECT username, email FROM users WHERE country='US' AND is_active=TRUE LIMIT 50\\\"
for tid, task in TASKS.items():
    s1, s2, s3 = task.grade(q)['score'], task.grade(q)['score'], task.grade(q)['score']
    assert s1 == s2 == s3, f'{tid} non-deterministic: {s1},{s2},{s3}'
\"" \
  "Graders must return same score for identical input every time"

# ── Summary ───────────────────────────────────────────────────────────────────
printf "\n${BOLD}══════════════════════════════════════════════════${NC}\n"
if [ "$FAIL" -eq 0 ]; then
  printf "${GREEN}${BOLD}  ✓ All %d checks passed — READY TO SUBMIT${NC}\n" "$PASS"
else
  printf "${RED}${BOLD}  ✗ %d/%d checks failed — fix before submitting${NC}\n" "$FAIL" "$((PASS+FAIL))"
fi
printf "${BOLD}══════════════════════════════════════════════════${NC}\n\n"

exit $FAIL