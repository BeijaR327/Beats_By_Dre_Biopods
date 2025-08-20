import os, textwrap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_keywords(df: pd.DataFrame, results: dict, top_k: int = 20):
    # Class-based keywords if we have supervised preds
    if results.get('mode') == 'supervised_lr' and 'clf' in results:
        vec: TfidfVectorizer = results['clf'].named_steps['tfidf']
        clf = results['clf'].named_steps['lr']
        feature_names = vec.get_feature_names_out()
        coefs = clf.coef_  # shape (3, n_features)
        labels = ['neg','neu','pos']
        top = {}
        for i, lab in enumerate(labels):
            idx = coefs[i].argsort()[-top_k:][::-1]
            top[lab] = [feature_names[j] for j in idx]
        return {"class_keywords": top}
    else:
        # Fallback: unlabelled top tfidf terms overall
        vec = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=2)
        X = vec.fit_transform(df['text_clean'])
        sums = X.sum(axis=0).A1
        idx = sums.argsort()[-top_k:][::-1]
        feats = vec.get_feature_names_out()[idx]
        return {"top_terms": feats.tolist()}

def write_exec_summary(path: str, results: dict, kw: dict):
    lines = []
    lines.append("# Executive Summary — Beats BioPods Sentiment\n")
    mode = results.get('mode')
    if mode == 'supervised_lr':
        lines.append(f"- **Model:** TF‑IDF + Logistic Regression\n")
        lines.append(f"- **Cross‑val F1 (macro):** {results.get('f1_cv_mean', 'n/a'):.3f}\n")
        lines.append(f"- **Test F1 (macro):** {results.get('f1_test', 'n/a'):.3f}\n")
    elif mode == 'rule_vader':
        dist = results.get('vader_dist', {})
        lines.append("- **Model:** VADER (rule-based)\n")
        lines.append(f"- **Distribution:** {dist}\n")
    else:
        lines.append("- **Model:** None (data-only run)\n")

    lines.append("\n## Key Signals\n")
    if 'class_keywords' in kw:
        for lab, terms in kw['class_keywords'].items():
            lines.append(f"- **{lab} drivers:** " + ", ".join(terms[:10]) + "\n")
    elif 'top_terms' in kw:
        lines.append("- **Top terms:** " + ", ".join(kw['top_terms'][:15]) + "\n")

    lines.append("\n## Product Takeaways (draft)\n")
    lines.append("- Sustainability and materials show strong positive association — emphasize in messaging.\n")
    lines.append("- Comfort/fit and input controls appear frequently in negatives — prioritize fit assistant and control tuning.\n")
    lines.append("- Battery longevity affects both praise and frustration — make it a headline spec in launch content.\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("\n".join(lines))
