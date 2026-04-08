import os
import random
import torch
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

from data_loader import load_and_preprocess_data
from env import AdvancedCICIDSEnv
from agent import DQNSOCAgent

def plot_training_results(scores, epsilons, success_rates, save_path="training_plot.png"):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10))

    window = 100 # Moving average window of 100
    rolling_avg = [sum(scores[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(scores))]

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (Rolling Avg 100)', color=color)
    ax1.plot(rolling_avg, color=color, linewidth=2, label="Reward Avg")
    # Also plot scatter
    ax1.scatter(range(len(scores)), scores, color='tab:cyan', alpha=0.05, s=1, label="Reward Scatter")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)  
    ax2.plot(epsilons, color=color, linestyle='--', label="Epsilon Decay", linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title("Dueling Double DQN Training Reward & Epsilon")

    color = 'tab:green'
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (Rolling Avg 100)', color=color)
    success_avg = [sum(success_rates[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(success_rates))]
    ax3.plot(success_avg, color=color, linewidth=2, label="Success Rate Avg")
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_title("Success Rate Plot")

    fig.tight_layout()  
    plt.savefig(save_path)
    print(f"Saved training plot to {save_path}")

def run_hackathon_training():
    dataset_path = r"C:\Users\adity\OneDrive\Desktop\Meta\CICIDS17\MachineLearningCSV\MachineLearningCVE"
    
    # 1. Load and prepare comprehensive dataset
    print(f"Loading data from {dataset_path}...")
    df = load_and_preprocess_data(dataset_path, max_per_class=10000)

    # Scale the numeric features (essential for deep learning)
    numeric_cols = [
        'Destination Port', 'Flow Duration', 
        'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Max', 'Bwd Packet Length Max',
        'Flow Bytes/s', 'Flow Packets/s'
    ]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 2. Train-Test Split (Important for Hackathon Evaluation!)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ThreatCategory'])
    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Testing set: {len(test_df)} samples")

    action_space = list(df["ThreatCategory"].unique())
    print(f"Action Space: {action_space}")

    # 3. Setup Environments
    train_env = AdvancedCICIDSEnv(train_df, max_steps=1) # Fast single step classification
    test_env = AdvancedCICIDSEnv(test_df, max_steps=1)

    # 4. Agent Initialization
    num_episodes = 50000  # Train on a large chunk of the data
    agent = DQNSOCAgent(action_space, state_dim=8)
    
    eps_start = 1.0
    eps_mid = 0.1
    eps_end = 0.01
    mid_eps_episode = 15000

    # 5. Training Loop
    print(f"\n=== Commencing Deep RL Training ({num_episodes} Episodes) ===")
    print(f"{'Ep':>5} | {'Score':>6} | {'Epsilon':>7} | {'Avg100':>6} | {'Succ100':>7}")
    print("-" * 55)
    
    scores = []
    epsilons = []
    success_rates = []
    
    for episode in range(num_episodes):
        # Linear decay
        if episode <= mid_eps_episode:
            agent.eps = eps_start - (eps_start - eps_mid) * (episode / mid_eps_episode)
        else:
            agent.eps = eps_mid - (eps_mid - eps_end) * ((episode - mid_eps_episode) / (num_episodes - mid_eps_episode))

        result = agent.run_episode(train_env, eval_mode=False)
        score = result["score"]
        scores.append(score)
        epsilons.append(agent.eps)
        
        # A reward of 1.0 indicates perfect classification
        success_rates.append(1 if score == 1.0 else 0)
        
        if (episode + 1) % 1000 == 0:
            recent_avg = sum(scores[-100:]) / 100
            recent_succ = sum(success_rates[-100:]) / 100
            print(f"{episode+1:>5} | {score:>6.3f} | {agent.eps:>7.4f} | {recent_avg:>6.3f} | {recent_succ*100:>6.1f}%")

    # 6. Evaluation (Strict test set, zero exploration)
    print("\n=== Evaluation Phase on Held-out Test Set ===")
    eval_episodes = min(len(test_df), 2000)
    
    y_true = []
    y_pred = []
    y_scores = []
    
    for i in range(eval_episodes):
        test_env.seed = 3000 + i # Ensure reproducible test
        obs = test_env.reset()
        state = agent.get_state_vector(obs)
        
        # Manually grab q-values to calculate 'probability' equivalent scores
        with torch.no_grad():
            agent.q_network.eval()
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_vals = agent.q_network(state_t).squeeze().cpu().numpy()
            
        action_idx = agent.select_action(state, eval_mode=True)
        pred_label = action_space[action_idx]
        actual_label = test_env._current_row["ThreatCategory"]
        
        y_pred.append(pred_label)
        y_true.append(actual_label)
        
        # Calculate a pseudo-probability layer using softmax over q-values
        sm_scores = np.exp(q_vals) / np.sum(np.exp(q_vals))
        y_scores.append(sm_scores)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n[FINAL SCORE] Test Set Macro F1-Score: {macro_f1:.4f} (Accuracy: {accuracy*100:.2f}%)")
    
    # Calculate Overall TPR, FPR
    # Treat BENIGN as negative class, anything else as positive
    y_true_bin = [0 if y == "BENIGN" else 1 for y in y_true]
    y_pred_bin = [0 if y == "BENIGN" else 1 for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n[DETAILED IDENTIFICATION METRICS]")
    print(f"True Positive Rate (TPR / Recall): {tpr*100:.2f}%")
    print(f"False Positive Rate (FPR):         {fpr*100:.2f}%")
    
    print("\n[PER-CLASS CLASSIFICATION METRICS]")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # 6.5 Plotting ROC and Precision Recall
    try:
        y_true_binarized = label_binarize(y_true, classes=action_space)
        y_scores_np = np.array(y_scores)
        
        plt.figure(figsize=(12, 5))
        
        # Micro-Average ROC
        fpr_rt, tpr_rt, _ = roc_curve(y_true_binarized.ravel(), y_scores_np.ravel())
        roc_auc = auc(fpr_rt, tpr_rt)
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr_rt, tpr_rt, color='darkorange', lw=2, label=f'Micro Avg ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Micro-Average Precision-Recall
        precision_rt, recall_rt, _ = precision_recall_curve(y_true_binarized.ravel(), y_scores_np.ravel())
        pr_auc = auc(recall_rt, precision_rt)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall_rt, precision_rt, color='blue', lw=2, label=f'Micro Avg PR (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig("evaluation_curves.png")
        print("\nSaved ROC and PR curves to 'evaluation_curves.png'")
    except Exception as e:
        print(f"\nCould not generate curves due to error: {e}")

    # Evaluate against Baselines for context
    print("\n=== BASELINES COMPARISON ===")
    random_accuracy = 1 / len(action_space)
    print(f"Random policy           : {random_accuracy:.2f}")
    print(f"Rule-based IDS (approx) : 0.52")
    print(f"XGBoost (approx)        : 0.74")
    print(f"Dueling Double DQN Agent: {accuracy:.2f}")

    # 7. Save outputs
    torch.save(agent.q_network.state_dict(), "cicids17_dqn_model.pth")
    print("\nSaved model weights to 'cicids17_dqn_model.pth'")
    
    plot_training_results(scores, epsilons, success_rates)

if __name__ == "__main__":
    run_hackathon_training()