from __future__ import annotations
"""
schemas.py - Pydantic v2 models for the Autonomous SOC Analyst OpenEnv environment.
"""

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
    action_type: ActionType = Field(..., description="The type of action the agent wants to take")
    target: Optional[str] = Field(None, description="Target IP address or hostname for the action")
    details: Optional[str] = Field(None, description="Additional details, e.g. alert message")
    final_answer: Optional[str] = Field(None, description="Final answer string when action_type is submit")


class Observation(BaseModel):
    task_id: str = Field(..., description="The current task identifier")
    step: int = Field(..., description="Current step number in the episode")
    logs: str = Field(..., description="JSON string of log entries available to the agent")
    blocked_ips: List[str] = Field(default_factory=list, description="List of IPs blocked so far")
    isolated_hosts: List[str] = Field(default_factory=list, description="List of hosts isolated so far")
    alerts_raised: List[str] = Field(default_factory=list, description="List of alerts raised so far")
    done: bool = Field(..., description="Whether the episode has ended")
    info: str = Field(default="", description="Additional information or feedback from the environment")


class Reward(BaseModel):
    value: float = Field(..., description="Scalar reward value in [0.0, 1.0]")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Detailed breakdown of reward layers")
    step: int = Field(..., description="Step number at which this reward was computed")


class EnvironmentState(BaseModel):
    task_id: str = Field(..., description="Current task identifier")
    step: int = Field(..., description="Current step count")
    done: bool = Field(..., description="Whether the episode is done")
    logs: str = Field(..., description="JSON string of all log entries for this episode")
    blocked_ips: List[str] = Field(default_factory=list, description="IPs currently blocked")
    isolated_hosts: List[str] = Field(default_factory=list, description="Hosts currently isolated")
    alerts_raised: List[str] = Field(default_factory=list, description="Alerts raised in this episode")
    score: float = Field(default=0.0, description="Current episode score")
    last_action_type: Optional[str] = Field(
        None, description="The action_type of the previous step, used for repeat-penalty detection"
    )
    repeat_action_count: int = Field(
        default=0,
        description="How many consecutive times the same invalid action has been repeated"
    )


class TaskSpec(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    description: str = Field(..., description="Natural language description of the task objective")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, or hard")
    max_steps: int = Field(..., description="Maximum number of steps allowed per episode")
    reward_threshold: float = Field(..., description="Minimum grader score to consider task solved")
