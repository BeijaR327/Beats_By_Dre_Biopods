import pandas as pd

def load_feedback(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("Expected a 'text' column in the input CSV.")
    # Normalize expected optional columns
    for c in ['channel','created_at','rating','label']:
        if c not in df.columns:
            df[c] = None
    return df
