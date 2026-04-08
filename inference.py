import os
import sys
import asyncio
import logging
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env import AdvancedCICIDSEnv
from schemas import Action, ActionType
from data_loader import load_and_preprocess_data
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def run_eval():
    baseUrl = os.getenv('API_BASE_URL', 'https://api.openai.com/v1')
    model = os.getenv('MODEL_NAME', 'gpt-4o-mini')
    api_key = os.getenv('OPENAI_API_KEY', 'dummy')

    client = AsyncOpenAI(base_url=baseUrl, api_key=api_key)
    
    df = load_and_preprocess_data("hf", max_per_class=200)
    env = AdvancedCICIDSEnv(df)

    tasks = ['cicids_easy', 'cicids_medium', 'cicids_hard']
    
    for _, task_id in enumerate(tasks):
        rewards = []
        
        logger.info(f"--- starting eval for {task_id} using {model} ---")
        try:
            obs = env.reset(task_id=task_id)
            
            for step in range(1, env.max_steps + 1):
                if env._done: break
                    
                try:
                    res = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a SOC analyst determining traffic anomalies. Output EXACTLY one category: BENIGN, DDOS, PORT_SCAN, WEBATTACK, INFILTRATION."},
                            {"role": "user", "content": f"task: {task_id}\nlogs: {obs.logs}\ncategory:"}
                        ],
                        max_tokens=10,
                        temperature=0.0
                    )
                    pred = (res.choices[0].message.content or "BENIGN").strip().upper()
                except Exception as exc:
                    logger.warning(f"failed to fetch from llm: {exc}")
                    pred = "BENIGN"
                    
                obs, reward, done, _ = env.step(Action(action_type=ActionType.submit, final_answer=pred))
                rewards.append(reward.value)
                
                logger.info(f"eval step={step} action={pred} rew={reward.value} done={done}")
                if done: break
                    
            final_score = sum(rewards) / len(rewards) if rewards else 0.0
            final_score = min(max(final_score, 0.0), 1.0)
            
            logger.info(f"finished task={task_id} score={final_score:.2f} \n")
            
        except Exception as e:
            logger.error(f"fatal env error: {e}")

if __name__ == "__main__":
    asyncio.run(run_eval())
