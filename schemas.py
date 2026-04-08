from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    query_logs = "query_logs"
    block_ip = "block_ip"
    isolate_host = "isolate_host"
    raise_alert = "raise_alert"
    submit = "submit"

class Action(BaseModel):
    action_type: ActionType
    target: Optional[str] = None
    details: Optional[str] = None
    final_answer: Optional[str] = None

class Observation(BaseModel):
    task_id: str
    step: int
    logs: str
    blocked_ips: List[str] = Field(default_factory=list)
    isolated_hosts: List[str] = Field(default_factory=list)
    alerts_raised: List[str] = Field(default_factory=list)
    done: bool
    info: str = ""

class Reward(BaseModel):
    value: float
    breakdown: Dict[str, Any] = Field(default_factory=dict)
    step: int

class EnvironmentState(BaseModel):
    task_id: str
    step: int
    done: bool
    logs: str
    blocked_ips: List[str] = Field(default_factory=list)
    isolated_hosts: List[str] = Field(default_factory=list)
    alerts_raised: List[str] = Field(default_factory=list)
    score: float = 0.0
    last_action_type: Optional[str] = None
    repeat_action_count: int = 0

class TaskSpec(BaseModel):
    task_id: str
    description: str
    difficulty: str
    max_steps: int
    reward_threshold: float
