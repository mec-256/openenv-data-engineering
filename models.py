from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal


class State(BaseModel):
    """The internal, hidden state of the environment."""

    task_id: str
    step_count: int = 0
    max_steps: int = 20
    is_done: bool = False
    cumulative_reward: float = 0.0
    tables_loaded: List[str] = Field(
        default_factory=list, description="Which CSV tables are currently loaded"
    )


class Observation(BaseModel):
    """The visible state of the data to the agent."""

    dataset_sample: List[Dict[str, Any]] = Field(
        ..., description="A sample of the current dataset"
    )
    columns: List[str] = Field(..., description="List of columns in the dataset")
    dtypes: Dict[str, str] = Field(..., description="Data types of columns")
    total_rows: int = Field(..., description="Total number of rows in the dataset")
    missing_values: Dict[str, int] = Field(
        ..., description="Count of missing values per column"
    )
    task_description: str = Field(
        ..., description="Instructions on what needs to be fixed"
    )
    feedback: str = Field(..., description="Feedback from the last action")
    debug_hints: List[str] = Field(
        default_factory=list,
        description="Optional debugging signals from the environment",
    )


class DropColumn(BaseModel):
    action_type: Literal["drop_column"] = "drop_column"
    column_name: str


class RenameColumn(BaseModel):
    action_type: Literal["rename_column"] = "rename_column"
    old_name: str
    new_name: str


class FillNaN(BaseModel):
    action_type: Literal["fill_nan"] = "fill_nan"
    column_name: str
    fill_value: Union[str, int, float, bool]


class ExtractRegex(BaseModel):
    action_type: Literal["extract_regex"] = "extract_regex"
    column_name: str
    regex_pattern: str
    new_column_names: List[str]


class CastType(BaseModel):
    action_type: Literal["cast_type"] = "cast_type"
    column_name: str
    target_type: Literal["string", "integer", "float", "boolean", "datetime"]


class DropDuplicates(BaseModel):
    action_type: Literal["drop_duplicates"] = "drop_duplicates"
    subset: Optional[List[str]] = None


class ParseJSONColumn(BaseModel):
    action_type: Literal["parse_json"] = "parse_json"
    column_name: str


class MergeTables(BaseModel):
    action_type: Literal["merge_tables"] = "merge_tables"
    left_on: str
    right_on: str
    how: Literal["inner", "left", "right", "outer"] = "inner"


class ExecutePandasCode(BaseModel):
    action_type: Literal["execute_pandas"] = "execute_pandas"
    code: str = Field(
        ...,
        description="Pandas code to execute. The dataset is available as 'df'. The last line must assign back to 'df'.",
    )


class Submit(BaseModel):
    action_type: Literal["submit"] = "submit"


Action = Union[
    DropColumn,
    RenameColumn,
    FillNaN,
    ExtractRegex,
    CastType,
    DropDuplicates,
    ParseJSONColumn,
    MergeTables,
    ExecutePandasCode,
    Submit,
]


class Reward(BaseModel):
    """Reward information returned after an action."""

    value: float = Field(
        ..., ge=-1.0, le=1.0, description="Reward value between -1.0 and 1.0"
    )
    message: str = Field(..., description="Explanation of the reward")
