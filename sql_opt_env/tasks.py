from dataclasses import dataclass
from typing import Dict, List

from .models import ColumnSchema, TableSchema


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split()).upper()


def contains_any(query: str, patterns: List[str]) -> bool:
    upper = normalize_query(query)
    return any(pattern in upper for pattern in patterns)


@dataclass
class Task:
    id: str
    difficulty: str
    description: str
    schemas: List[TableSchema]
    original_query: str
    original_cost: float
    max_steps: int = 10
    success_threshold: float = 0.7

    def grade(self, query: str) -> Dict[str, float]:
        normalized = normalize_query(query)

        if not normalized.startswith(("SELECT", "WITH")):
            return {"score": 0.0}

        if self.id == "select_star_cleanup":
            score = 0.0
            if "SELECT *" not in normalized:
                score += 0.4
            if "IS_ACTIVE=TRUE" in normalized or "IS_ACTIVE = TRUE" in normalized:
                score += 0.3
            if any(column in normalized for column in ["USERNAME", "EMAIL", "ID"]):
                score += 0.3
            return {"score": min(score, 1.0)}

        if self.id == "n_plus_one_to_join":
            score = 0.1
            if "JOIN" in normalized:
                score += 0.4
            if "GROUP BY" in normalized:
                score += 0.3
            if not contains_any(normalized, ["IN (SELECT", "EXISTS (SELECT"]):
                score += 0.2
            return {"score": min(score, 1.0)}

        if self.id == "window_function_rewrite":
            score = 0.1
            if contains_any(normalized, ["OVER (", "ROW_NUMBER()", "RANK()", "DENSE_RANK()"]):
                score += 0.5
            if "PARTITION BY" in normalized or "GROUP BY" in normalized:
                score += 0.2
            if not contains_any(normalized, ["IN (SELECT", "EXISTS (SELECT"]):
                score += 0.2
            return {"score": min(score, 1.0)}

        return {"score": 0.0}

    def issues(self, query: str, previous_query: str) -> List[str]:
        normalized = normalize_query(query)
        issues: List[str] = []

        if not normalized.startswith(("SELECT", "WITH")):
            issues.append("Query must start with SELECT or WITH.")
            return issues

        if self.id == "select_star_cleanup":
            if "SELECT *" in normalized:
                issues.append("SELECT * is still present; use explicit columns.")
            if "IS_ACTIVE=TRUE" not in normalized and "IS_ACTIVE = TRUE" not in normalized:
                issues.append("Missing is_active=TRUE filter on the users table.")
            if normalize_query(query) == normalize_query(self.original_query):
                issues.append("The query is unchanged from the original.")

        if self.id == "n_plus_one_to_join":
            if contains_any(normalized, ["IN (SELECT", "EXISTS (SELECT"]):
                issues.append("The correlated subquery remains; convert it to a JOIN or CTE.")
            if "JOIN" not in normalized and "WITH" not in normalized:
                issues.append("Use JOIN or CTE to avoid N+1 execution patterns.")
            if "GROUP BY" not in normalized:
                issues.append("Group results by user_id to aggregate orders efficiently.")

        if self.id == "window_function_rewrite":
            if not contains_any(normalized, ["OVER (", "ROW_NUMBER()", "RANK()", "DENSE_RANK()"]):
                issues.append("Add a window function such as ROW_NUMBER() or OVER() to avoid nested iteration.")
            if contains_any(normalized, ["IN (SELECT", "EXISTS (SELECT"]):
                issues.append("A correlated subquery is still present; replace it with a set-based window rewrite.")
            if "PARTITION BY" not in normalized and "GROUP BY" not in normalized:
                issues.append("Use PARTITION BY or GROUP BY to compute aggregates in one pass.")

        if normalize_query(query) == normalize_query(previous_query) and query.strip():
            issues.append("The current query is identical to the previous one.")

        return issues

    def hints(self, query: str, step: int) -> List[str]:
        if step > 3:
            return []

        if self.id == "select_star_cleanup":
            return [
                "Avoid SELECT * and list only the needed columns.",
                "Keep the is_active=TRUE filter to preserve business constraints.",
            ]

        if self.id == "n_plus_one_to_join":
            return [
                "Rewrite the correlated subquery using JOIN and GROUP BY.",
                "Keep the user_id relationship intact to preserve correctness.",
            ]

        if self.id == "window_function_rewrite":
            return [
                "Use a window function with PARTITION BY to compute aggregates efficiently.",
                "Avoid correlated subqueries by rewriting them as set operations.",
            ]

        return []


def simulate_cost(query: str, task: Task) -> float:
    normalized = normalize_query(query)
    cost = max(1.0, task.original_cost)

    if "SELECT *" in normalized:
        cost *= 1.4
    if contains_any(normalized, ["IN (SELECT", "EXISTS (SELECT"]):
        cost *= 1.3
    if "JOIN" in normalized:
        cost *= 0.9
    if "GROUP BY" in normalized:
        cost *= 0.85
    if contains_any(normalized, ["OVER (", "ROW_NUMBER()", "RANK()", "DENSE_RANK()"]):
        cost *= 0.75

    grade = task.grade(query)["score"]
    cost *= max(0.15, 1.0 - 0.7 * grade)
    cost += 0.01 * len(normalized)
    return max(1.0, round(cost, 2))


def build_schema(name: str, description: str, row_count: int, columns: List[Dict]) -> TableSchema:
    return TableSchema(name=name, description=description, row_count=row_count, columns=[ColumnSchema(**c) for c in columns])


TASKS: Dict[str, Task] = {
    "select_star_cleanup": Task(
        id="select_star_cleanup",
        difficulty="easy",
        description="Replace SELECT * with specific columns and preserve is_active filtering for the users table.",
        schemas=[
            build_schema(
                name="users",
                description="Active user directory with profile and status metadata.",
                row_count=1000000,
                columns=[
                    {"name": "id", "type": "INTEGER", "indexed": True, "description": "Primary user identifier."},
                    {"name": "username", "type": "TEXT", "indexed": False, "description": "User login name."},
                    {"name": "email", "type": "TEXT", "indexed": False, "description": "User email address."},
                    {"name": "country", "type": "TEXT", "indexed": True, "description": "User country code."},
                    {"name": "is_active", "type": "BOOLEAN", "indexed": True, "description": "Active account flag."},
                ],
            )
        ],
        original_query="SELECT * FROM users",
        original_cost=20000.0,
    ),
    "n_plus_one_to_join": Task(
        id="n_plus_one_to_join",
        difficulty="medium",
        description="Rewrite a correlated subquery into a JOIN-based aggregation over orders and users.",
        schemas=[
            build_schema(
                name="users",
                description="User directory table for customer metadata.",
                row_count=1000000,
                columns=[
                    {"name": "id", "type": "INTEGER", "indexed": True, "description": "Primary key for users."},
                    {"name": "username", "type": "TEXT", "indexed": False, "description": "User login name."},
                    {"name": "country", "type": "TEXT", "indexed": True, "description": "Country code for filtering."},
                ],
            ),
            build_schema(
                name="orders",
                description="Customer orders table with foreign key to users.",
                row_count=50000000,
                columns=[
                    {"name": "id", "type": "INTEGER", "indexed": True, "description": "Order identifier."},
                    {"name": "user_id", "type": "INTEGER", "indexed": True, "description": "Foreign key to users."},
                    {"name": "total", "type": "NUMERIC", "indexed": False, "description": "Order total amount."},
                ],
            ),
        ],
        original_query="SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE country = 'US')",
        original_cost=1500000.0,
    ),
    "window_function_rewrite": Task(
        id="window_function_rewrite",
        difficulty="hard",
        description="Rewrite a nested aggregation into a window function over orders and order_items.",
        schemas=[
            build_schema(
                name="orders",
                description="Order header table with customer and timestamp metadata.",
                row_count=50000000,
                columns=[
                    {"name": "id", "type": "INTEGER", "indexed": True, "description": "Order identifier."},
                    {"name": "user_id", "type": "INTEGER", "indexed": True, "description": "Referencing the user who placed the order."},
                    {"name": "status", "type": "TEXT", "indexed": False, "description": "Order status."},
                ],
            ),
            build_schema(
                name="order_items",
                description="Line items for orders with product and quantity details.",
                row_count=200000000,
                columns=[
                    {"name": "order_id", "type": "INTEGER", "indexed": True, "description": "Foreign key to orders."},
                    {"name": "product_id", "type": "INTEGER", "indexed": False, "description": "Product reference."},
                    {"name": "quantity", "type": "INTEGER", "indexed": False, "description": "Item quantity."},
                ],
            ),
        ],
        original_query="SELECT order_id, SUM(quantity) FROM order_items WHERE order_id IN (SELECT id FROM orders WHERE status = 'completed') GROUP BY order_id",
        original_cost=1638400000.0,
    ),
}
