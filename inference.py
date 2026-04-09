import os
import sys
import asyncio
from typing import List, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env import AdvancedCICIDSEnv
from schemas import Action, ActionType
from data_loader import load_and_preprocess_data
from openai import AsyncOpenAI

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)

async def main():
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("HF_TOKEN")

    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    
    df = load_and_preprocess_data("hf", max_per_class=200)
    env = AdvancedCICIDSEnv(df)

    tasks = ['cicids_easy', 'cicids_medium', 'cicids_hard']
    
    for task_id in tasks:
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False
        
        log_start(task=task_id, env="Autonomous SOC Analyst", model=model_name)
        
        try:
            obs = env.reset(task_id=task_id)
            
            for step_num in range(1, env.max_steps + 1):
                if env._done: break
                    
                error_msg = None
                try:
                    res = await client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a SOC analyst determining traffic anomalies. First you MUST output 'QUERY' to extract deep packet stats if they are missing. Then, to make a final decision, output 'SUBMIT <CATEGORY>' where CATEGORY is one of: BENIGN, DOS, PORTSCAN, WEBATTACK, INFILTRATION, BRUTEFORCE, BOTNET."},
                            {"role": "user", "content": f"task: {task_id}\nlogs: {obs.logs}\naction:"}
                        ],
                        max_tokens=15,
                        temperature=0.0
                    )
                    pred = (res.choices[0].message.content or "SUBMIT BENIGN").strip().upper()
                except Exception as exc:
                    error_msg = str(exc)
                    pred = "SUBMIT BENIGN"
                    
                if "QUERY" in pred:
                    action = Action(action_type=ActionType.query_logs)
                else:
                    cat = pred.split()[-1] if " " in pred else pred.replace("SUBMIT", "").strip() or "BENIGN"
                    action = Action(action_type=ActionType.submit, final_answer=cat)

                obs, reward, done, _ = env.step(action)
                rewards.append(reward.value)
                steps_taken = step_num
                
                log_step(step=step_num, action=pred, reward=reward.value, done=done, error=error_msg)
                
                if done: break
                    
            if rewards:
                score = sum(rewards) / len(rewards)
                score = min(max(score, 0.01), 0.99)
            success = score >= 0.5
            
        except Exception as e:
            print(f"[DEBUG] fatal env error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
