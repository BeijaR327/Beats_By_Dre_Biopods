from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

def _ensure_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

def train_or_rule_sentiment(df: pd.DataFrame, cfg: Dict):
    use_vader = cfg.get('use_vader', True)
    use_supervised = cfg.get('use_supervised', True) and 'label' in df.columns and df['label'].notna().any()
    result = {"mode": None}

    if use_supervised:
        # Map labels to numeric
        label_map = {'neg': 0, 'neu': 1, 'pos': 2}
        y = df['label'].map(label_map)
        mask = y.notna()
        X = df.loc[mask, 'text_clean']
        y = y[mask].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.get('test_size', 0.2), random_state=cfg.get('random_state', 42), stratify=y
        )

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ('lr', LogisticRegression(max_iter=1000))
        ])
        cv = StratifiedKFold(n_splits=cfg.get('cv_folds', 5), shuffle=True, random_state=42)
        f1_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')

        result.update({
            "mode": "supervised_lr",
            "f1_test": float(f1),
            "f1_cv_mean": float(np.mean(f1_cv)),
            "clf": pipe,
            "label_map": {v:k for k,v in label_map.items()},
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        })
        # Also produce vader on full df for comparison
        if use_vader:
            _ensure_vader()
            sia = SentimentIntensityAnalyzer()
            df['vader_compound'] = df['text_clean'].apply(lambda s: sia.polarity_scores(s)['compound'])
    else:
        result["mode"] = "rule_vader" if use_vader else "none"
        if use_vader:
            _ensure_vader()
            sia = SentimentIntensityAnalyzer()
            df['vader_compound'] = df['text_clean'].apply(lambda s: sia.polarity_scores(s)['compound'])
            # discrete mapping for reporting
            df['vader_label'] = pd.cut(df['vader_compound'], bins=[-1, -0.05, 0.05, 1], labels=['neg','neu','pos'])
            result["vader_dist"] = df['vader_label'].value_counts(normalize=True).to_dict()

    return result
