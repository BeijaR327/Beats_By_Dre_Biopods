import argparse, os, json
import pandas as pd
from src.data_load import load_feedback
from src.preprocess import build_text_pipeline
from src.model import train_or_rule_sentiment
from src.evaluate import evaluate_and_plot
from src.insights import summarize_keywords, write_exec_summary
import yaml

def main(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    paths = cfg['paths']
    df = load_feedback(paths['input_csv'])
    text_col = 'text'

    # Build preprocessing pipeline and transform
    pipe = build_text_pipeline(cfg.get('preprocess', {}))
    df['text_clean'] = pipe.transform(df[text_col])

    # Train/evaluate
    results = train_or_rule_sentiment(df, cfg.get('sentiment', {}))
    figs_dir = paths['figs_dir']
    os.makedirs(figs_dir, exist_ok=True)
    evaluate_and_plot(df, results, figs_dir)

    # Keywords / topics (class-based TF-IDF, noun phrases)
    kw = summarize_keywords(df, results, top_k=cfg['keywords']['top_k'])

    # Summary
    os.makedirs(os.path.dirname(paths['summary_md']), exist_ok=True)
    write_exec_summary(paths['summary_md'], results, kw)

    print("Done. Figures in:", figs_dir)
    print("Summary:", paths['summary_md'])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
