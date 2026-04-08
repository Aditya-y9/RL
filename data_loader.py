import os
import glob
import pandas as pd
import numpy as np

def load_and_preprocess_data(dataset_path: str, max_per_class: int = 5000):
    """
    Loads all CSV files from the CIC-IDS-2017 dataset folder.
    Cleans column names, handles NaNs/Infs, groups labels, and balances the dataset.
    """
    print(f"Loading CSVs from: {dataset_path}")
    all_csvs = glob.glob(os.path.join(dataset_path, "*.csv"))
    
    if not all_csvs:
        raise ValueError(f"No CSVs found in {dataset_path}")
        
    df_list = []
    
    # We only take features that are crucial and fast to train on for the SOC analyst
    columns_to_keep = [
        'Destination Port', 'Flow Duration', 
        'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Max', 'Bwd Packet Length Max',
        'Flow Bytes/s', 'Flow Packets/s', 'Label'
    ]
    
    for f in all_csvs:
        print(f"Reading {os.path.basename(f)}...")
        try:
            # Read only a subset of columns to save memory if possible
            df = pd.read_csv(f, engine='python', on_bad_lines='skip')
            # Clean column names (strip trailing/leading spaces)
            df.columns = df.columns.str.strip()
            
            # Select only the columns we want, ignore if some are missing
            available_cols = [c for c in columns_to_keep if c in df.columns]
            df = df[available_cols]
            
            # Clean numeric data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Sub-sample immediately to prevent massive memory usage
            # Take at most `max_per_class` from each file's classes
            for label, group in df.groupby('Label'):
                df_list.append(group.sample(n=min(len(group), max_per_class), random_state=42))

        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        raise ValueError("No data could be read.")
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Group similar attack classes into broader threat categories to simplify the RL action space
    def map_attack_labels(label):
        lbl = str(label).upper()
        if 'BENIGN' in lbl: return 'BENIGN'
        if 'DOS' in lbl or 'DDOS' in lbl or 'HULK' in lbl or 'SLOWLORIS' in lbl or 'GOLDENEYE' in lbl: 
            return 'DOS'
        if 'PORT' in lbl and 'SCAN' in lbl: 
            return 'PORTSCAN'
        if 'BRUTE' in lbl or 'PATATOR' in lbl: 
            return 'BRUTEFORCE'
        if 'BOT' in lbl: 
            return 'BOTNET'
        if 'WEB' in lbl or 'XSS' in lbl or 'SQL' in lbl: 
            return 'WEBATTACK'
        if 'INFILTRATION' in lbl:
            return 'INFILTRATION'
        return 'OTHER'
        
    full_df['ThreatCategory'] = full_df['Label'].apply(map_attack_labels)
    
    # Balance dataset across all loaded data up to max_per_class
    print("Balancing dataset...")
    balanced_chunks = []
    
    # Filter out categories that are too small to learn properly or just 'OTHER'
    full_df = full_df[full_df['ThreatCategory'] != 'OTHER']
    
    for category, group in full_df.groupby('ThreatCategory'):
        # Oversample if under max_per_class, otherwise sample exactly max_per_class
        n_samples = max_per_class
        balanced_chunks.append(group.sample(n=n_samples, replace=True, random_state=42))
        
    final_df = pd.concat(balanced_chunks).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("\nDataset preparation complete. Final distribution:")
    print(final_df['ThreatCategory'].value_counts())
    
    return final_df

if __name__ == "__main__":
    # Test loader
    path = r"C:\Users\adity\OneDrive\Desktop\Meta\CICIDS17\MachineLearningCSV\MachineLearningCVE"
    df = load_and_preprocess_data(path, max_per_class=1000)
    print(f"Total shape shape: {df.shape}")