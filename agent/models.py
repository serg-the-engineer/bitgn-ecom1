from __future__ import annotations

from typing import Annotated, Literal, Union

from annotated_types import Ge, Le, MaxLen, MinLen
from pydantic import BaseModel, Field


class ReportTaskCompletion(BaseModel):
    tool: Literal["report_completion"]
    completed_steps_laconic: list[str]
    message: str
    grounding_refs: list[str] = Field(default_factory=list)
    outcome: Literal[
        "OUTCOME_OK",
        "OUTCOME_DENIED_SECURITY",
        "OUTCOME_NONE_CLARIFICATION",
        "OUTCOME_NONE_UNSUPPORTED",
        "OUTCOME_ERR_INTERNAL",
    ]


class Req_Tree(BaseModel):
    tool: Literal["tree"]
    level: int = Field(2, description="max tree depth, 0 means unlimited")
    root: str = Field("", description="tree root, empty means repository root")


class Req_Find(BaseModel):
    tool: Literal["find"]
    name: str
    root: str = "/"
    kind: Literal["all", "files", "dirs"] = "all"
    limit: Annotated[int, Ge(1), Le(20)] = 10


class Req_Search(BaseModel):
    tool: Literal["search"]
    pattern: str
    limit: Annotated[int, Ge(1), Le(20)] = 10
    root: str = "/"


class Req_List(BaseModel):
    tool: Literal["list"]
    path: str = "/"


class Req_Read(BaseModel):
    tool: Literal["read"]
    path: str
    number: bool = Field(False, description="return 1-based line numbers")
    start_line: Annotated[int, Ge(0)] = Field(
        0, description="1-based inclusive line; 0 means from the first line"
    )
    end_line: Annotated[int, Ge(0)] = Field(
        0, description="1-based inclusive line; 0 means through the last line"
    )


class Req_Write(BaseModel):
    tool: Literal["write"]
    path: str
    content: str


class Req_Delete(BaseModel):
    tool: Literal["delete"]
    path: str


class Req_Stat(BaseModel):
    tool: Literal["stat"]
    path: str


class Req_Exec(BaseModel):
    tool: Literal["exec"]
    path: str
    args: list[str] = Field(default_factory=list)
    stdin: str = ""


AgentCommand = Union[
    ReportTaskCompletion,
    Req_Tree,
    Req_Find,
    Req_Search,
    Req_List,
    Req_Read,
    Req_Write,
    Req_Delete,
    Req_Stat,
    Req_Exec,
]


class NextStep(BaseModel):
    current_state: str
    plan_remaining_steps_brief: Annotated[list[str], MinLen(1), MaxLen(5)] = Field(
        ...,
        description="briefly explain the next useful steps",
    )
    decision_summary: str = Field(
        "",
        description="brief, safe, observable explanation for why this next action was selected",
    )
    uncertainty_flags: list[str] = Field(
        default_factory=list,
        description="explicit uncertainties or assumptions to monitor before acting",
    )
    task_completed: bool
    function: AgentCommand = Field(..., description="execute the first remaining step")


class AgentRunResult(BaseModel):
    final_outcome: str | None = None
    final_message: str | None = None
    completed_steps_laconic: list[str] = Field(default_factory=list)
    grounding_refs: list[str] = Field(default_factory=list)
    steps: int = 0
    stopped_reason: Literal["reported_completion", "step_limit", "exception"]
    exception: dict | None = None
    counters: dict = Field(default_factory=dict)
