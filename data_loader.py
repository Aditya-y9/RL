import os
import glob
import pandas as pd
import numpy as np

def load_and_preprocess_data(dataset_path: str, max_per_class: int = 5000):
    df_list = []
    
    columns_to_keep = [
        'Destination Port', 'Flow Duration', 
        'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Max', 'Bwd Packet Length Max',
        'Flow Bytes/s', 'Flow Packets/s', 'Label'
    ]
    
    def process_dataframe(df):
        df.columns = df.columns.str.strip()
        available_cols = [c for c in columns_to_keep if c in df.columns]
        df = df[available_cols].copy()
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        return [group.sample(n=min(len(group), max_per_class), random_state=42) 
                for _, group in df.groupby('Label')]
    
    if dataset_path == "hf":
        print("fetching c01dsnap/CIC-IDS2017 from hf hub")
        from datasets import load_dataset
        ds = load_dataset('c01dsnap/CIC-IDS2017', split='train')
        df_list.extend(process_dataframe(ds.to_pandas()))
            
    else:
        print(f"reading local csvs from {dataset_path}")
        all_csvs = glob.glob(os.path.join(dataset_path, "*.csv"))
        if not all_csvs:
            raise ValueError(f"no csv files found in {dataset_path}")
            
        for f in all_csvs:
            try:
                df = pd.read_csv(f, engine='python', on_bad_lines='skip')
                df_list.extend(process_dataframe(df))
            except Exception as e:
                print(f"skipping {f}: {e}")
            
    if not df_list:
        raise ValueError("could not read any data")
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    def map_attack_labels(label):
        lbl = str(label).upper()
        if 'BENIGN' in lbl: return 'BENIGN'
        if any(x in lbl for x in ['DOS', 'DDOS', 'HULK', 'SLOWLORIS', 'GOLDENEYE']): return 'DOS'
        if 'PORT' in lbl and 'SCAN' in lbl: return 'PORTSCAN'
        if 'BRUTE' in lbl or 'PATATOR' in lbl: return 'BRUTEFORCE'
        if 'BOT' in lbl: return 'BOTNET'
        if any(x in lbl for x in ['WEB', 'XSS', 'SQL']): return 'WEBATTACK'
        if 'INFILTRATION' in lbl: return 'INFILTRATION'
        return 'OTHER'
        
    full_df['ThreatCategory'] = full_df['Label'].apply(map_attack_labels)
    full_df = full_df[full_df['ThreatCategory'] != 'OTHER']
    
    balanced_chunks = [
        group.sample(n=max_per_class, replace=True, random_state=42)
        for _, group in full_df.groupby('ThreatCategory')
    ]
        
    final_df = pd.concat(balanced_chunks).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("\nclass distribution:")
    print(final_df['ThreatCategory'].value_counts())
    
    return final_df

if __name__ == "__main__":
    df = load_and_preprocess_data("hf", max_per_class=1000)
    print(df.shape)