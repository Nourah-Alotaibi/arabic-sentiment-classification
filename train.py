"""Deduplicated Arabic sentiment benchmark; training-only vocabularies."""

import argparse, re, unicodedata, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from common import *


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub("[\u064b-\u065f\u0670\u0640]", "", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def prepare(data, out):
    d = pd.read_excel(data) if str(data).endswith(".xlsx") else pd.read_csv(data)
    d = d[["Text", "Sentiment"]].dropna()
    n = len(d)
    d["Sentiment"] = d.Sentiment.astype(str).str.strip().str.lower()
    d = d[d.Sentiment.isin(["negative", "neutral", "positive"])].copy()
    d["normalized"] = d.Text.map(normalize)
    d = d[d.normalized.str.len() > 2]
    conflicts = d.groupby("normalized").Sentiment.nunique()
    ambiguous = set(conflicts[conflicts > 1].index)
    namb = int(d.normalized.isin(ambiguous).sum())
    d = (
        d[~d.normalized.isin(ambiguous)]
        .drop_duplicates("normalized")
        .reset_index(drop=True)
    )
    dev, test = train_test_split(
        np.arange(len(d)), test_size=0.2, stratify=d.Sentiment, random_state=SEED
    )
    tr, va = train_test_split(
        dev, test_size=0.25, stratify=d.iloc[dev].Sentiment, random_state=SEED
    )
    split = np.full(len(d), "train", dtype=object)
    split[va] = "validation"
    split[test] = "test"
    d["split"] = split
    # Local dataset is ignored by Git and contains no exported raw social-media rows in results.
    out = Path(out)
    private = out.parent / "data"
    private.mkdir(exist_ok=True)
    d.to_csv(private / "prepared.csv", index=False)
    audit = {
        "original_rows": n,
        "retained_unique_rows": len(d),
        "conflicting_label_rows_removed": namb,
        "removed_or_duplicate_rows": n - len(d),
        "class_counts": d.Sentiment.value_counts().to_dict(),
        "split_sizes": pd.Series(split).value_counts().to_dict(),
        "input_sha256": sha256(data),
    }
    save_json(out / "data_audit.json", audit)
    return d, dev, tr, va, test, audit


def run(data, out):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    start = time.time()
    d, dev, tr, va, test, audit = prepare(data, out)
    X = d.normalized
    y = d.Sentiment

    def word():
        return TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=70000
        )

    def char():
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=120000,
        )

    candidates = {
        "RBF-SVM baseline": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=1000)),
                ("classifier", SVC(C=1, kernel="rbf")),
            ]
        ),
        "Word TFIDF + LinearSVC": Pipeline(
            [("tfidf", word()), ("classifier", LinearSVC(C=1))]
        ),
        "Character TFIDF + LinearSVC": Pipeline(
            [("tfidf", char()), ("classifier", LinearSVC(C=1))]
        ),
        "Word+character TFIDF + LinearSVC": Pipeline(
            [
                ("features", FeatureUnion([("word", word()), ("char", char())])),
                ("classifier", LinearSVC(C=1)),
            ]
        ),
        "Word+character TFIDF + LogisticRegression": Pipeline(
            [
                ("features", FeatureUnion([("word", word()), ("char", char())])),
                ("classifier", LogisticRegression(C=3, max_iter=1000)),
            ]
        ),
    }
    rows = []
    for name, m in candidates.items():
        print("Fitting", name, flush=True)
        m.fit(X.iloc[tr], y.iloc[tr])
        rows.append({"model": name, **evaluate(y.iloc[va], m.predict(X.iloc[va]))})
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_comparison.csv", index=False)
    best = table.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    test_results = {}
    for name in dict.fromkeys(["RBF-SVM baseline", best]):
        m = candidates[name]
        m.fit(X.iloc[dev], y.iloc[dev])
        pred = m.predict(X.iloc[test])
        test_results[name] = evaluate(y.iloc[test], pred)
        test_results[name]["accuracy_95pct_wilson"] = accuracy_interval(
            y.iloc[test], pred
        )
        if name == best:
            classification_artifacts(y.iloc[test], pred, out)
            models = out.parent / "models"
            models.mkdir(exist_ok=True)
            joblib.dump(m, models / "model.joblib")
    compare_plot(table, "macro_f1", out / "validation_comparison.png")
    report = {
        "dataset": "Local arabic_tweets.xlsx; label provenance not independently verified",
        "audit": audit,
        "selected_model": best,
        "selection_metric": "validation macro F1",
        "test_results": test_results,
        "legacy_accuracy": 0.615614,
        "legacy_comparability": "Different deduplication and split; use same-split baseline for improvement claims.",
        "runtime_seconds": time.time() - start,
        "environment": versions(),
    }
    save_json(out / "metrics.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.data, a.out)
