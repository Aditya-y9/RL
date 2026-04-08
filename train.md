# Explanation of `train.py`

This file coordinates loading the data, creating the environment, initializing the agent, running the main deep reinforcement learning loop, and evaluating the final artifact.

*   Imports include basic utilities (`os`, `random`, `json`), PyTorch (`torch`), Scikit-Learn (`train_test_split`, `metrics`, `preprocessing`, `StandardScaler`), Matplotlib handles (`pyplot`), Pandas, and custom classes (`load_and_preprocess_data`, `AdvancedCICIDSEnv`, `DQNSOCAgent`).

### `plot_training_results`
*   `def plot_training_results(scores, epsilons, success_rates, save_path="training_plot.png"):`: A helper function using `matplotlib` to plot two graphs:
    *   The top subplot shows the raw simulation rewards and moving average over the episodes spanning epsilon's decay on an alternate axis.
    *   The bottom subplot plots the rolling 100-episode success (perfect detection rate).
    *   Saves the resulting plot directly into `training_plot.png`.

### `run_hackathon_training`
*   `def run_hackathon_training():`: The main script routine orchestrating the entire RL lifecycle.
*   **Data Preparation**:
    *   Invokes `load_and_preprocess_data` aiming for tens of thousands of rows spanning different attack classes.
    *   Performs `StandardScaler().fit_transform(...)` to normalize inputs to mean 0, variance 1. High variation otherwise destabilizes neural networks.
    *   Splits the standardized data via `train_test_split` into 80% train and 20% independent test targets.
*   **Setup**: Submits the filtered DataFrames into train and evaluate variants of the `AdvancedCICIDSEnv` mapped to one step each. Creates the agent `DQNSOCAgent` against the recognized classes.
*   **Training Loop**:
    *   Loops for `num_episodes` times.
    *   Forces $\epsilon$ decay linearly across pre-defined breakpoints for controlled exploration.
    *   `agent.run_episode(train_env, eval_mode=False)` prompts the episode simulation. It appends scoring to logs and performs `print()` formatting across 1000-episode boundaries.
*   **Evaluation Phase**:
    *   Runs on up to 2000 episodes on the purely unseen test split.
    *   Employs `agent.select_action(eval_mode=True)` shutting off exploration. It also manually pulls $Q$-values mimicking softmax probability behaviors to graph Receiver Operating Characteristic (ROC) or Precision Recall sets.
    *   Performs comprehensive SOC analysis: Accuracy, macro F1, exact Precision/Recall stats for individual specific classes (BOTNET vs BRUTEvs ...).
    *   Attempts to plot robust ROC curve using mathematical metrics per subset prediction against true subset prediction.
*   **Baselines and Disk Commit**:
    *   Prints standard naive-ML comparisons (like guessing algorithms and standard XGBOOST behaviors) and contrasts them.
    *   Saves the model dictionary to `cicids17_dqn_model.pth`.
    *   Fires off the aforementioned plots.

### `if __name__ == "__main__":`
*   Triggers the `run_hackathon_training()` function if the script runs directly.
