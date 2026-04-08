import os
import sys
import asyncio
from typing import List, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pydantic import BaseModel
from env import AdvancedCICIDSEnv
from schemas import Action, ActionType
from data_loader import load_and_preprocess_data
import pandas as pd
from openai import AsyncOpenAI, OpenAI

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)


    

async def main() -> None:
    api_base_url = os.getenv('API_BASE_URL', 'https://api.openai.com/v1')
    model_name = os.getenv('MODEL_NAME', 'gpt-4o-mini')
    api_key = os.getenv('HF_TOKEN', 'dummy-token')

    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    
    df = load_and_preprocess_data("dataset_sample/", max_per_class=200)
    env = AdvancedCICIDSEnv(df)

    tasks = ['task_1_easy', 'task_2_medium', 'task_3_hard']
    categories = ['BENIGN', 'DDOS', 'PORT_SCAN']
    
    for i, task_id in enumerate(tasks):
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        
        log_start(task=task_id, env="Autonomous SOC Analyst", model=model_name)
        
        try:
            obs = env.reset(task_id=task_id)
            
            for step in range(1, env.max_steps + 1):
                if env._done:
                    break
                    
                # Actual call to OpenAI model as required by the contest rules
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are an AI SOC Analyst. Classify the traffic. Reply ONLY with one of: BENIGN, DDOS, PORT_SCAN, WEBATTACK, INFILTRATION."},
                            {"role": "user", "content": f"Task: {task_id}\nLogs: {obs.logs}\nClassify the threat category:"}
                        ],
                        max_tokens=10,
                        temperature=0.0
                    )
                    prediction = (response.choices[0].message.content or "BENIGN").strip().upper()
                except Exception as exc:
                    print(f"[DEBUG] Model request failed: {exc}", flush=True)
                    prediction = "BENIGN"

                    
                action_msg = prediction
                action = Action(action_type=ActionType.submit, final_answer=prediction)
                
                obs, reward, done, info = env.step(action)
                
                val = reward.value
                error = None
                
                rewards.append(val)
                steps_taken = step
                
                log_step(step=step, action=action_msg, reward=val, done=done, error=error)
                
                history.append(f"Step {step}: {action_msg!r} -> reward {val:+.2f}")
                
                if done:
                    break
                    
            score = sum(rewards) / len(rewards) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= 0.5
            
        except Exception as e:
            print(f"[DEBUG] env error: {e}", flush=True)
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
