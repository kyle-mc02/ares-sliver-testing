# Trains a Random Forest on one C2 tool's traffic and evaluates on another,
# using the same GridSearchCV + StratifiedKFold pipeline as Parssegny et al.

# Must be run from the src/ directory so that utils/ and ml_config_detailed.toml
# are on the path.
import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import toml
import os
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    make_scorer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../src")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from utils.estimator_builder import (
        get_estimator,
        build_grid_search_cv_classifier_pipeline,
    )
except ImportError as e:
    print(f"ERROR: Could not import utils.estimator_builder: {e}")
    print("  Ensure the script is in the ares-sliver-testing/ directory")
    sys.exit(1)


def run_experiment(
    train_path: str,
    train_label: str,
    test_path: str,
    test_label: str,
    ml_config: dict,
    n_jobs: int,
    seed: int,
    description: str,
) -> dict:
    # Train on train_path, evaluate on test_path.
    # Both CSVs must have a 'label' column.
    # train_label / test_label are the positive (malicious) class strings.
    # Returns a dict of results.


    # Load data
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    if "label" not in train_df.columns:
        raise ValueError(f"No 'label' column in {train_path}")
    if "label" not in test_df.columns:
        raise ValueError(f"No 'label' column in {test_path}")

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test  = test_df.drop("label", axis=1)
    y_test  = test_df["label"]

    # Verify labels exist
    if train_label not in y_train.unique():
        raise ValueError(f"train-label '{train_label}' not found in {train_path}. "
                         f"Available: {y_train.unique().tolist()}")
    if test_label not in y_test.unique():
        raise ValueError(f"test-label '{test_label}' not found in {test_path}. "
                         f"Available: {y_test.unique().tolist()}")

    # Build pipeline (identical to learning_curve_build.py)
    param_grid = ml_config["RFC_PARAMS"]
    scoring = {
        "f1":        make_scorer(f1_score,        pos_label=train_label),
        "precision": make_scorer(precision_score,  pos_label=train_label),
        "recall":    make_scorer(recall_score,     pos_label=train_label),
    }

    clf = get_estimator("rf", random_state=seed)
    estimator = build_grid_search_cv_classifier_pipeline(
        classifier=clf,
        parameter_d=param_grid,
        n_jobs=n_jobs,
        scoring=scoring,
        n_splits=10,
        refit="f1",
        random_state=seed,
    )

    # Train 
    estimator.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = estimator.predict(X_test)

    f1  = f1_score(y_test, y_pred,        pos_label=test_label, zero_division=0)
    pre = precision_score(y_test, y_pred,  pos_label=test_label, zero_division=0)
    rec = recall_score(y_test, y_pred,     pos_label=test_label, zero_division=0)

    cm = confusion_matrix(
        y_test, y_pred,
        labels=[test_label, next(l for l in y_test.unique() if l != test_label)]
    )

    train_mal   = int((y_train == train_label).sum())
    train_benign = int((y_train != train_label).sum())
    test_mal    = int((y_test  == test_label).sum())
    test_benign = int((y_test  != test_label).sum())

    return {
        "description":    description,
        "train_path":     train_path,
        "test_path":      test_path,
        "train_label":    train_label,
        "test_label":     test_label,
        "train_malicious": train_mal,
        "train_benign":   train_benign,
        "test_malicious": test_mal,
        "test_benign":    test_benign,
        "best_params":    estimator.best_params_,
        "f1":             round(f1,  4),
        "precision":      round(pre, 4),
        "recall":         round(rec, 4),
        "confusion_matrix": cm.tolist(),
        "timestamp":      datetime.now().isoformat(),
    }

def print_results(r: dict):
    print(f"\n{'='*60}")
    print(f"  {r['description']}")
    print(f"{'='*60}")
    print(f"  Train: {r['train_path']}")
    print(f"         {r['train_malicious']} malicious ({r['train_label']}), "
          f"{r['train_benign']} benign")
    print(f"  Test:  {r['test_path']}")
    print(f"         {r['test_malicious']} malicious ({r['test_label']}), "
          f"{r['test_benign']} benign")
    print(f"\n  Best RF params: {r['best_params']}")
    print(f"\n  F1:        {r['f1']:.3f}")
    print(f"  Precision: {r['precision']:.3f}")
    print(f"  Recall:    {r['recall']:.3f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  [TP  FN]   {r['confusion_matrix'][0]}")
    print(f"  [FP  TN]   {r['confusion_matrix'][1]}")
    print()

def main():
    ap = argparse.ArgumentParser(
        description="Cross-tool C2 detection experiment using Parssegny RF pipeline."
    )
    ap.add_argument("--train",       required=True, help="Path to training CSV")
    ap.add_argument("--train-label", required=True, help="Malicious class label in training CSV")
    ap.add_argument("--test",        required=True, help="Path to test CSV")
    ap.add_argument("--test-label",  required=True, help="Malicious class label in test CSV")
    ap.add_argument("--desc",        default="Cross-tool experiment",
                    help="Description label for output")
    ap.add_argument("--both",        action="store_true",
                    help="Also run the reverse experiment (swap train and test)")
    ap.add_argument("--config",      default="ml_config_detailed.toml",
                    help="Path to Parssegny ML config TOML (default: ml_config_detailed.toml)")
    ap.add_argument("--jobs",  "-j", type=int, default=4, help="Parallel jobs for GridSearchCV")
    ap.add_argument("--seed",        type=int, default=0,  help="Random seed")
    ap.add_argument("--output", "-o", default=None,
                    help="Optional path to write JSON results file")
    args = ap.parse_args()

    ml_config = toml.load(args.config)
    all_results = []

    # Forward experiment
    r1 = run_experiment(
        train_path=args.train,
        train_label=args.train_label,
        test_path=args.test,
        test_label=args.test_label,
        ml_config=ml_config,
        n_jobs=args.jobs,
        seed=args.seed,
        description=args.desc,
    )
    print_results(r1)
    all_results.append(r1)

    # Reverse experiment
    if args.both:
        reverse_desc = f"{args.desc} (reverse)"
        r2 = run_experiment(
            train_path=args.test,
            train_label=args.test_label,
            test_path=args.train,
            test_label=args.train_label,
            ml_config=ml_config,
            n_jobs=args.jobs,
            seed=args.seed,
            description=reverse_desc,
        )
        print_results(r2)
        all_results.append(r2)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<35} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['description']:<35} {r['f1']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f}")
    print()

    # JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results written to {args.output}")


if __name__ == "__main__":
    main()