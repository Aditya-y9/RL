import json
import random
import sys
import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# Ensure schemas module can be imported from parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from schemas import EnvironmentState, Observation, Action, ActionType, Reward

class AdvancedCICIDSEnv:
    def __init__(self, df: pd.DataFrame, max_steps: int = 15, seed: int = 42):
        """
        OpenEnv-compliant environment for CICIDS-17 Threat Classification.
        """
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
        """OpenEnv standard state inspection method."""
        return self._state

    def reset(self, task_id: Optional[str] = None) -> Observation:
        row_idx = self._rng.randint(0, len(self.df) - 1)
        self._current_row = self.df.iloc[row_idx]
        
        self._step_count = 0
        self._done = False
        
        log_json = {
            "Destination_Port": float(self._current_row.get("Destination Port", 0)),
            "Flow_Duration": float(self._current_row.get("Flow Duration", 0)),
            "Total_Fwd_Packets": float(self._current_row.get("Total Fwd Packets", 0)),
            "Total_Bwd_Packets": float(self._current_row.get("Total Backward Packets", 0)),
            "Fwd_Packet_Length_Max": float(self._current_row.get("Fwd Packet Length Max", 0)),
            "Bwd_Packet_Length_Max": float(self._current_row.get("Bwd Packet Length Max", 0)),
            "Flow_Bytes_s": float(self._current_row.get("Flow Bytes/s", 0)),
            "Flow_Packets_s": float(self._current_row.get("Flow Packets/s", 0))
        }
        
        self._state = EnvironmentState(
            task_id="cicids_advanced_classification",
            step=0,
            done=False,
            logs=json.dumps([log_json]),
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
            
            is_prediction_attack = (prediction != "BENIGN")
            is_truth_attack = (ground_truth != "BENIGN")
            
            if prediction == ground_truth:
                reward_val = 1.0  # Correct precise classification (Max 1.0)
                details = f"Correct classification: {ground_truth}"
            else:
                if not is_truth_attack and is_prediction_attack:
                    # False Alarm (Predicted Attack, was Benign) - explicitly heavily penalized
                    reward_val = 0.0
                    details = f"False alarm. Predicted {prediction}"
                elif ground_truth in ["WEBATTACK", "INFILTRATION"] and not is_prediction_attack:
                    reward_val = 0.0  # Penalize missing critical attacks
                    details = f"Severe missed attack! Actual {ground_truth}"
                elif is_truth_attack and not is_prediction_attack:
                    # Missed Attack (Predicted Benign, was Attack)
                    reward_val = 0.0
                    details = f"Missed attack. Actual {ground_truth}"
                elif is_truth_attack and is_prediction_attack and prediction != ground_truth:
                    # Partial Mitigation (Detected an attack, but got the wrong category)
                    reward_val = 0.5
                    details = f"Partial detection. Predicted {prediction}, Actual {ground_truth}"
                else:
                    reward_val = 0.0
                    details = f"Incorrect. Predicted {prediction}, Actual {ground_truth}"
                
            self._done = True
        elif action.action_type == ActionType.query_logs:
            # Querying consumes a step and a small time penalty (kept as 0 in output but stored in details)
            reward_val = 0.0
            details = "Extracted more logs."
        else:
            reward_val = 0.0
            details = "Action not supported in this threat classification task snippet."
            
        if self._step_count >= self.max_steps:
            self._done = True
                
        self._state.step = self._step_count
        self._state.done = self._done
        self._state.score += reward_val
        
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