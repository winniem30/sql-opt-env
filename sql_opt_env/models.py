from typing import List, Optional

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    name: str
    type: str
    indexed: bool = False
    description: Optional[str] = None


class TableSchema(BaseModel):
    name: str
    description: str
    row_count: int
    columns: List[ColumnSchema]


class SQLOptObservation(BaseModel):
    task_id: str
    difficulty: str
    description: str
    schemas: List[TableSchema]
    original_query: str
    current_query: str
    step_number: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    syntax_valid: bool
    execution_error: Optional[str] = None
    estimated_cost: float
    original_cost: float
    cost_improvement_ratio: Optional[float] = None
    issues_found: List[str] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)
    last_reward: float = Field(ge=0.0, le=1.0)
    cumulative_reward: float = Field(ge=0.0)
    query_plan: Optional[str] = None
    rows_returned: Optional[int] = None


class SQLOptAction(BaseModel):
    query: str
    explanation: Optional[str] = None


class SQLOptReward(BaseModel):
    reward: float = Field(ge=0.0, le=1.0)
    grade_score: float = Field(ge=0.0, le=1.0)
    cost_improvement: float
    done: bool
