"""
main.py - master script.   Usage:  python main.py [--retrain] [--no-dashboard]

 validate project -> load data -> prepare -> split/SMOTE/scale -> train (or load saved
 models) -> evaluate -> save to models/ -> launch the Streamlit dashboard.
"""
import argparse
import subprocess
import sys

import pandas as pd

import pipeline as P


def first_csv(folder):
    files = sorted(folder.glob("*.csv"))
    return files[0] if files else None


def load_validated(path, need_target):
    df = pd.read_csv(path)
    problems = P.validate_raw_dataset(df, need_target)
    if problems:
        sys.exit(f"ERROR in {path.name}:\n  - " + "\n  - ".join(problems))
    print(f"Loaded {path.name}: {len(df):,} rows x {df.shape[1]} columns")
    return df


def run_training(models):
    train_path = first_csv(P.TRAIN_DIR)
    if train_path is None:
        sys.exit(f"ERROR: no training CSV found in {P.TRAIN_DIR}. Put the raw train.csv there.")
    train = P.prepare_data(load_validated(train_path, need_target=True))

    test_prepared, test_path = None, first_csv(P.TEST_DIR)
    if test_path is not None:   # optional separate, labelled test file
        test_prepared = P.prepare_data(load_validated(test_path, need_target=True), drop_zero_var=False)
    else:
        print("No file in data/testing - using the notebook's held-out 20% split as the test set.")
    P.train_and_save(train, test_prepared, models)


def launch_dashboard():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit("Streamlit is not installed. Run: pip install -r requirements.txt")
    print("Launching dashboard ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(P.ROOT / "dashboard.py")], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="retrain even if saved models exist")
    ap.add_argument("--models", default=",".join(P.MODEL_NAMES), help="comma list: " + ",".join(P.MODEL_NAMES))
    ap.add_argument("--no-dashboard", action="store_true")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    bad = [m for m in models if m not in P.MODEL_NAMES]
    if bad:
        sys.exit(f"Unknown model(s) {bad}. Choose from {list(P.MODEL_NAMES)}")

    saved = all((P.MODEL_DIR / f"{m}.joblib").exists() for m in models)
    if saved and not args.retrain:
        print("Saved models found in models/ - loading them instead of retraining (use --retrain to override).")
    elif first_csv(P.TRAIN_DIR) is not None:
        run_training(models)
    else:
        print("No saved models and no training CSV in data/training - the dashboard will ask you to upload them.")
    if not args.no_dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
