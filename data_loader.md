# Explanation of `data_loader.py`

This file is responsible for reading the CIC-IDS-2017 datase from CSV files, preprocessing it, separating categories, and balancing the dataset.

*   `import os`, `import glob`: Used for handling paths and scanning directories for files.
*   `import pandas as pd`, `import numpy as np`: Used for data manipulation, mathematical operations, and DataFrame handling.

### `load_and_preprocess_data`
*   `def load_and_preprocess_data(dataset_path: str, max_per_class: int = 5000):`: Takes a path to the directory with CSV files and a cap on how many samples to take per class.
*   `all_csvs = glob.glob(os.path.join(dataset_path, "*.csv"))`: Finds all CSV files in the input directory.
*   `columns_to_keep = [...]`: A hardcoded list of the most critical and relevant numerical features to extract from the datasets, plus the target `Label`.
*   The `for f in all_csvs:` loop iterates over all found files:
    *   `df = pd.read_csv(f, engine='python', on_bad_lines='skip')`: Reads each CSV, skipping problematic lines.
    *   `df.columns = df.columns.str.strip()`: Cleans up any trailing whitespace in column names.
    *   `df = df[available_cols]`: Filters down only to the predefined `columns_to_keep` that exist in `df`.
    *   `df.replace([np.inf, -np.inf], np.nan, inplace=True)`: Replaces infinities with `NaN`.
    *   `df.dropna(inplace=True)`: Drops rows containing missing values (`NaN`).
    *   The internal subset sampling takes up to `max_per_class` rows from each class in that specific file to manage memory.
*   `full_df = pd.concat(df_list, ignore_index=True)`: Joins all chunk databaframe pieces together into one DataFrame.
*   `def map_attack_labels(label):`: A helper function that clusters the detailed dataset labels into broader attack categories (e.g., merging all DoS types into just "DOS").
*   `full_df['ThreatCategory'] = full_df['Label'].apply(map_attack_labels)`: Creates the simplified target feature based on our helper function.
*   The final balancing step attempts to sample exactly `max_per_class` for every `ThreatCategory` (oversampling or undersampling as necessary), drops the "OTHER" category, and thoroughly shuffles the rows with `.sample(frac=1.0)`.
*   `return final_df`: Returns the strictly numeric, balanced, and cleaned DataFrame ready for ML usage.

### `if __name__ == "__main__":`
*   A test block ensuring the `load_and_preprocess_data` function successfully targets the dataset path and outputs the shape of the resulting DataFrame.
