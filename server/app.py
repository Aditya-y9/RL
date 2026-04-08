import os
import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env import AdvancedCICIDSEnv
from data_loader import load_and_preprocess_data
from schemas import Action, ActionType, Observation, EnvironmentState

app = FastAPI()

# Global env instance
df = load_and_preprocess_data("dataset_sample/", max_per_class=200)
env_instance = AdvancedCICIDSEnv(df)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Autonomous SOC Analyst OpenEnv is running! API paths: /reset, /step, /state"}

@app.post("/reset")
def reset_endpoint(task_id: str = None):
    obs = env_instance.reset(task_id=task_id)
    return {"observation": obs.dict()}

@app.post("/step")
def step_endpoint(action: Action):
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state_endpoint():
    return env_instance.state().dict()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()