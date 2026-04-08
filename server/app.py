import os
import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env import AdvancedCICIDSEnv
from data_loader import load_and_preprocess_data
from schemas import Action

app = FastAPI()

df = load_and_preprocess_data("hf", max_per_class=5000)
env = AdvancedCICIDSEnv(df)

@app.get("/")
def health_check():
    return {
        "status": "ok", 
        "msg": "soc analyst openenv running",
        "endpoints": ["/reset", "/step", "/state"]
    }

@app.post("/reset")
def reset_env(task_id: str = None):
    return {"observation": env.reset(task_id=task_id).dict()}

@app.post("/step")
def step_env(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    return env.state().dict()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()