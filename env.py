import json
import random
import sys
import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from schemas import EnvironmentState, Observation, Action, ActionType, Reward

class AdvancedCICIDSEnv:
    def __init__(self, df: pd.DataFrame, max_steps: int = 15, seed: int = 42):
        self.df = df
        self.seed = seed
        self._rng = random.Random(seed)
        
        self.classes = list(df["ThreatCategory"].unique())
        self.max_steps = max_steps
        
        self._state = None
        self._done = False
        self._step_count = 0
        self._current_row = None

    def state(self) -> EnvironmentState:
        return self._state

    def reset(self, task_id: Optional[str] = None) -> Observation:
        task_id = task_id or "task_1_easy"
        
        if "easy" in task_id.lower():
            valid_cats = ["BENIGN", "DOS"]
        elif "medium" in task_id.lower():
            valid_cats = ["BENIGN", "PORTSCAN", "BRUTEFORCE", "BOTNET"]
        elif "hard" in task_id.lower():
            valid_cats = ["BENIGN", "WEBATTACK", "INFILTRATION"]
        else:
            valid_cats = self.classes
            
        filtered_df = self.df[self.df["ThreatCategory"].isin(valid_cats)]
        if len(filtered_df) == 0: filtered_df = self.df
            
        row_idx = self._rng.randint(0, len(filtered_df) - 1)
        self._current_row = filtered_df.iloc[row_idx]
        
        self._step_count = 0
        self._done = False
        
        initial_logs = {
            "Destination_Port": float(self._current_row.get("Destination Port", 0)),
            "Flow_Duration": float(self._current_row.get("Flow Duration", 0)),
            "Total_Fwd_Packets": float(self._current_row.get("Total Fwd Packets", 0)),
            "Total_Bwd_Packets": float(self._current_row.get("Total Backward Packets", 0))
        }
        
        self._hidden_logs = {
            "Fwd_Packet_Length_Max": float(self._current_row.get("Fwd Packet Length Max", 0)),
            "Bwd_Packet_Length_Max": float(self._current_row.get("Bwd Packet Length Max", 0)),
            "Flow_Bytes_s": float(self._current_row.get("Flow Bytes/s", 0)),
            "Flow_Packets_s": float(self._current_row.get("Flow Packets/s", 0))
        }
        
        self._state = EnvironmentState(
            task_id=task_id,
            step=0,
            done=False,
            logs=json.dumps([initial_logs]),
            score=0.0
        )
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._step_count += 1
        reward_val = 0.0
        details = ""
        
        if action.action_type == ActionType.submit:
            prediction = action.final_answer
            ground_truth = self._current_row["ThreatCategory"]
            
            is_pred_attack = prediction != "BENIGN"
            is_true_attack = ground_truth != "BENIGN"
            
            if prediction == ground_truth:
                reward_val = 0.99
                details = f"correct: {ground_truth}"
            elif not is_true_attack and is_pred_attack:
                reward_val = 0.01
                details = f"false alarm, was benign"
            elif ground_truth in ["WEBATTACK", "INFILTRATION"] and not is_pred_attack:
                reward_val = 0.01
                details = f"missed critical attack: {ground_truth}"
            elif is_true_attack and not is_pred_attack:
                reward_val = 0.01
                details = f"missed attack: {ground_truth}"
            elif is_true_attack and is_pred_attack and prediction != ground_truth:
                reward_val = 0.5
                details = f"partial hit (pred: {prediction}, true: {ground_truth})"
            else:
                reward_val = 0.01
                details = f"wrong (pred: {prediction}, true: {ground_truth})"
                
            self._done = True
            
        elif action.action_type == ActionType.query_logs:
            reward_val = 0.01
            details = "extracted deep packet stats"
            
            if hasattr(self, '_hidden_logs') and self._hidden_logs:
                current_logs = json.loads(self._state.logs)
                current_logs[0].update(self._hidden_logs)
                self._state.logs = json.dumps(current_logs)
                self._hidden_logs = {}
            
        else:
            reward_val = 0.01
            details = "invalid action"
            
        if self._step_count >= self.max_steps:
            self._done = True
                
        self._state.step = self._step_count
        self._state.done = self._done
        self._state.score += reward_val
        
        if self._done:
            self._state.score = min(max(self._state.score, 0.01), 0.99)
        else:
            self._state.score = min(max(self._state.score, 0.01), 0.99)
        
        obs = self._make_observation(info=details)
        reward = Reward(value=reward_val, breakdown={"details": details}, step=self._step_count)
        
        return obs, reward, self._done, {"ground_truth": self._current_row["ThreatCategory"]}

    def _make_observation(self, info: str = "") -> Observation:
        return Observation(
            task_id=self._state.task_id,
            step=self._state.step,
            logs=self._state.logs,
            done=self._state.done,
            info=info
        )