"""preprocessing.py"""
import pandas as pd
import numpy as np
from typing import Optional


def load_csv(path: str) -> pd.DataFrame:
    """Read CSV into DataFrame (thin wrapper so callers can mock in tests)."""
    return pd.read_csv(path)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 -> float32 and int64 -> int32 where safe to reduce memory."""
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_integer_dtype(df[col]):
            # skip if contains NA
            if df[col].isnull().any():
                continue
            df[col] = df[col].astype('int32')
    return df


def preprocess(df: pd.DataFrame, drop_cols: Optional[list] = None) -> pd.DataFrame:
    """Main preprocessing pipeline. Returns processed DataFrame (not written to disk here).
    
    - drops requested columns
    - maps binary fields
    - expands comma-separated drug columns into dummies
    - fills missing values with 0 for modeling convenience (caller can change)
    """
    df = df.copy()
    
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    
    # mapping examples (defensive: check existence)
    if 'Patient_Smoker' in df.columns:
        df['Patient_Smoker'] = df['Patient_Smoker'].map({'YES': 1, 'NO': 0, 'Cannot say': np.nan})
    
    if 'Patient_Rural_Urban' in df.columns:
        df['Patient_Rural_Urban'] = df['Patient_Rural_Urban'].map({'RURAL': 1, 'URBAN': 0})
    
    if 'Treated_with_drugs' in df.columns:
        # split comma-separated drug lists into dummies
        dummies = df['Treated_with_drugs'].str.get_dummies(sep=',')
        df = pd.concat([df.drop(columns=['Treated_with_drugs']), dummies], axis=1)
    
    # Drop rows where crucial label is missing (caller can choose otherwise)
    if 'Survived_1_year' in df.columns:
        df = df.dropna(subset=['Survived_1_year'])
    
    # Fill other missing values with 0 by default
    df = df.fillna(0)
    
    df = optimize_dtypes(df)
    return df


def save_processed(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # small CLI for quick local runs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', default='processed_dataset.csv')
    args = parser.parse_args()
    df_raw = load_csv(args.infile)
    df_proc = preprocess(df_raw, drop_cols=['ID_Patient_Care_Situation', 'Patient_ID'])
    save_processed(df_proc, args.outfile)
    print(f"Saved processed dataset to {args.outfile}")