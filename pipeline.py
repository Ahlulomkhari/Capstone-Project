"""
pipeline.py - the notebook's pipeline, extracted into functions.

Everything here is lifted from FINALLL.ipynb (CRISP-DM sections 3-5) with the
same logic, split ratios, SMOTE, scaling, hyper-parameter grids and threshold
search. Only print() calls were removed and a bundle format for saving was added.
"""
import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:  # notebook text calls the fallback "XGBoost-equivalent (HGB)"
    xgb = None

ROOT = Path(__file__).resolve().parent
MODEL_DIR, TRAIN_DIR, TEST_DIR = ROOT / "models", ROOT / "data" / "training", ROOT / "data" / "testing"
FULL_DIR = ROOT / "data" / "full"   # optional: the complete labelled dataset for the dashboard Overview
RANDOM_STATE = 42
TARGET = "LOAN_DEFAULT"
LABEL_MAP = {0: "No Default", 1: "Default"}
MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest",
               "logistic_regression": "Logistic Regression"}

ID_COLUMNS_TO_DROP = ["UNIQUEID", "BRANCH_ID", "SUPPLIER_ID", "MANUFACTURER_ID",
                      "CURRENT_PINCODE_ID", "STATE_ID", "EMPLOYEE_CODE_ID"]

# Raw columns the notebook's prepare_data() reads (IDs are dropped if present, not required).
REQUIRED_RAW_COLUMNS = [
    "DATE_OF_BIRTH", "DISBURSAL_DATE", "AVERAGE_ACCT_AGE", "CREDIT_HISTORY_LENGTH",
    "PERFORM_CNS_SCORE_DESCRIPTION", "EMPLOYMENT_TYPE", "DISBURSED_AMOUNT", "ASSET_COST",
    "PRI_NO_OF_ACCTS", "SEC_NO_OF_ACCTS", "PRI_ACTIVE_ACCTS", "SEC_ACTIVE_ACCTS",
    "PRI_OVERDUE_ACCTS", "SEC_OVERDUE_ACCTS", "PRI_CURRENT_BALANCE", "SEC_CURRENT_BALANCE",
    "PRI_SANCTIONED_AMOUNT", "SEC_SANCTIONED_AMOUNT", "PRI_DISBURSED_AMOUNT",
    "SEC_DISBURSED_AMOUNT", "PRIMARY_INSTAL_AMT", "SEC_INSTAL_AMT",
    "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS", "AADHAR_FLAG", "PAN_FLAG", "VOTERID_FLAG",
    "DRIVING_FLAG", "PASSPORT_FLAG"]

RISK_CATEGORY_MAP = {
    "No Bureau History Available": "No History/Not Scored",
    "Not Scored: Sufficient History Not Available": "No History/Not Scored",
    "Not Scored: Not Enough Info available on the customer": "No History/Not Scored",
    "Not Scored: No Activity seen on the customer (Inactive)": "No History/Not Scored",
    "Not Scored: No Updates available in last 36 months": "No History/Not Scored",
    "Not Scored: Only a Guarantor": "No History/Not Scored",
    "Not Scored: More than 50 active Accounts found": "No History/Not Scored",
    "M-Very High Risk": "High", "L-Very High Risk": "High", "K-High Risk": "High", "J-High Risk": "High",
    "I-Medium Risk": "Medium", "H-Medium Risk": "Medium",
    "G-Low Risk": "Low", "F-Low Risk": "Low", "E-Low Risk": "Low",
    "D-Very Low Risk": "Low", "C-Very Low Risk": "Low", "B-Very Low Risk": "Low", "A-Very Low Risk": "Low"}


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_raw_dataset(df, need_target):
    """Return a list of human-readable problems (empty list = OK)."""
    problems = []
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): {missing}")
    if need_target:
        if TARGET not in df.columns:
            problems.append(f"Target column '{TARGET}' is missing.")
        elif df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
            problems.append(f"'{TARGET}' must contain only 0/1 with no blanks.")
    if len(df) < 50:
        problems.append(f"Only {len(df)} rows - too few to train or evaluate.")
    return problems


def validate_model_features(bundle, columns):
    """Compare what the model expects with what the prepared data provides."""
    expected = list(bundle["feature_cols"])
    missing = [c for c in expected if c not in columns]
    extra = [c for c in columns if c not in expected]
    n_in = getattr(bundle["model"], "n_features_in_", len(expected))
    msgs = []
    if n_in != len(expected):
        msgs.append(f"Model reports {n_in} input features but its feature list has {len(expected)}.")
    if missing:
        msgs.append(f"Data is missing {len(missing)} feature(s) the model needs: {missing}")
    return (not msgs), missing, extra, msgs


# ----------------------------------------------------------------------
# Data preparation (notebook section 3, unchanged logic)
# ----------------------------------------------------------------------
def duration_to_months(s):
    if pd.isna(s):
        return np.nan
    yrs, mon = 0, 0
    parts = s.replace("yrs", "").replace("mon", "").split()
    if len(parts) == 2:
        yrs, mon = int(parts[0]), int(parts[1])
    return yrs * 12 + mon


def prepare_data(data, drop_zero_var=True):
    """Full notebook prepare_data(). Set drop_zero_var=False for inference/test files
    so a column that happens to be constant in a small file is not dropped."""
    data = data.copy()
    data = data.drop(columns=[c for c in ID_COLUMNS_TO_DROP if c in data.columns])

    # 3.2 dates
    data["DATE_OF_BIRTH"] = pd.to_datetime(data["DATE_OF_BIRTH"], format="%d-%m-%Y")
    data["DISBURSAL_DATE"] = pd.to_datetime(data["DISBURSAL_DATE"], format="%d-%m-%Y")
    data["AGE"] = (data["DISBURSAL_DATE"] - data["DATE_OF_BIRTH"]).dt.days // 365
    data.loc[(data["AGE"] <= 0) | (data["AGE"] > 100), "AGE"] = np.nan
    data["AGE"] = data["AGE"].fillna(data["AGE"].mean())
    data["DISBURSAL_MONTH"] = data["DISBURSAL_DATE"].dt.month
    data = data.drop(columns=["DATE_OF_BIRTH", "DISBURSAL_DATE"])
    data["AVERAGE_ACCT_AGE_MONTHS"] = data["AVERAGE_ACCT_AGE"].apply(duration_to_months)
    data["CREDIT_HISTORY_LENGTH_MONTHS"] = data["CREDIT_HISTORY_LENGTH"].apply(duration_to_months)
    data = data.drop(columns=["AVERAGE_ACCT_AGE", "CREDIT_HISTORY_LENGTH"])

    # 3.3 bureau risk category
    data["CREDIT_RISK_CATEGORY"] = data["PERFORM_CNS_SCORE_DESCRIPTION"].map(RISK_CATEGORY_MAP)
    data["HAS_BUREAU_SCORE"] = (data["CREDIT_RISK_CATEGORY"] != "No History/Not Scored").astype(int)
    data = data.drop(columns=["PERFORM_CNS_SCORE_DESCRIPTION"])

    # 3.4 encoding
    data["EMPLOYMENT_TYPE"] = data["EMPLOYMENT_TYPE"].map({"Salaried": 1, "Self employed": 0})
    data["CREDIT_RISK_GRADE"] = data["CREDIT_RISK_CATEGORY"].map(
        {"No History/Not Scored": 0, "High": 1, "Medium": 2, "Low": 3})

    # 3.5 RF imputation of EMPLOYMENT_TYPE (uses features only, never the target)
    cols = [c for c in data.columns
            if c not in ["EMPLOYMENT_TYPE", TARGET, "CREDIT_RISK_CATEGORY"] and data[c].dtype != "object"]
    known, unknown = data[data["EMPLOYMENT_TYPE"].notna()], data[data["EMPLOYMENT_TYPE"].isna()]
    if len(unknown) > 0 and len(known) > 0:
        imp = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
        imp.fit(known[cols], known["EMPLOYMENT_TYPE"])
        data.loc[data["EMPLOYMENT_TYPE"].isna(), "EMPLOYMENT_TYPE"] = imp.predict(unknown[cols])

    # 3.6 ratios / aggregates
    d = data
    d["LOAN_TO_ASSET_RATIO"] = d["DISBURSED_AMOUNT"] / d["ASSET_COST"]
    d["TOTAL_ACCTS"] = d["PRI_NO_OF_ACCTS"] + d["SEC_NO_OF_ACCTS"]
    d["TOTAL_ACTIVE_ACCTS"] = d["PRI_ACTIVE_ACCTS"] + d["SEC_ACTIVE_ACCTS"]
    d["TOTAL_OVERDUE_ACCTS"] = d["PRI_OVERDUE_ACCTS"] + d["SEC_OVERDUE_ACCTS"]
    d["TOTAL_CURRENT_BALANCE"] = d["PRI_CURRENT_BALANCE"] + d["SEC_CURRENT_BALANCE"]
    d["TOTAL_SANCTIONED_AMOUNT"] = d["PRI_SANCTIONED_AMOUNT"] + d["SEC_SANCTIONED_AMOUNT"]
    d["TOTAL_DISBURSED_AMOUNT_BUREAU"] = d["PRI_DISBURSED_AMOUNT"] + d["SEC_DISBURSED_AMOUNT"]
    d["TOTAL_INSTALLMENT_AMT"] = d["PRIMARY_INSTAL_AMT"] + d["SEC_INSTAL_AMT"]
    d["INSTALLMENT_TO_DISBURSED_RATIO"] = (
        d["PRIMARY_INSTAL_AMT"] / d["DISBURSED_AMOUNT"].replace(0, np.nan)).fillna(0)
    d["DELINQUENCY_RATE"] = (
        d["DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS"] / d["TOTAL_ACCTS"].replace(0, np.nan)).fillna(0)
    d["ID_DOCS_PROVIDED"] = d[["AADHAR_FLAG", "PAN_FLAG", "VOTERID_FLAG",
                               "DRIVING_FLAG", "PASSPORT_FLAG"]].sum(axis=1)
    redundant = ["VOTERID_FLAG", "PRI_DISBURSED_AMOUNT", "PRI_SANCTIONED_AMOUNT", "PRI_NO_OF_ACCTS",
                 "PRI_ACTIVE_ACCTS", "PRI_OVERDUE_ACCTS", "PRI_CURRENT_BALANCE", "PRIMARY_INSTAL_AMT",
                 "SEC_NO_OF_ACCTS", "SEC_ACTIVE_ACCTS", "SEC_OVERDUE_ACCTS", "SEC_CURRENT_BALANCE",
                 "SEC_SANCTIONED_AMOUNT", "SEC_DISBURSED_AMOUNT", "SEC_INSTAL_AMT"]
    d = d.drop(columns=[c for c in redundant if c in d.columns])
    if drop_zero_var:
        d = d.drop(columns=[c for c in d.columns if d[c].nunique(dropna=False) <= 1])
    return d


def feature_matrix(prepared):
    """Notebook: X = data.drop(['LOAN_DEFAULT', 'CREDIT_RISK_CATEGORY'])."""
    return prepared.drop(columns=[c for c in [TARGET, "CREDIT_RISK_CATEGORY"] if c in prepared.columns])


def split_data(prepared, separate_test=False):
    """Notebook split: 60/20/20 stratified. If a separate test file is supplied,
    only the train/val split (75/25 of the training file) is needed."""
    y, X = prepared[TARGET], feature_matrix(prepared)
    if separate_test:
        return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25,
                                                random_state=RANDOM_STATE, stratify=y_tmp)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def smote_oversample_minority(X_train, y_train, k_neighbors=5, random_state=RANDOM_STATE):
    """Notebook's custom SMOTE - training set only."""
    rng = np.random.RandomState(random_state)
    X_arr, y_arr = X_train.values.astype(float), y_train.values
    counts = pd.Series(y_arr).value_counts()
    minority, n_gen = counts.idxmin(), counts.max() - counts.min()
    X_min = X_arr[y_arr == minority]
    k = min(k_neighbors, len(X_min) - 1)
    _, nbr = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min)
    synth = np.zeros((n_gen, X_arr.shape[1]))
    for i in range(n_gen):
        b = rng.randint(0, len(X_min))
        n = nbr[b][rng.randint(1, k + 1)]
        synth[i] = X_min[b] + rng.rand() * (X_min[n] - X_min[b])
    Xb = pd.DataFrame(np.vstack([X_arr, synth]), columns=X_train.columns)
    yb = pd.Series(np.concatenate([y_arr, np.full(n_gen, minority)]), name=y_train.name).astype(int)
    return Xb, yb


# ----------------------------------------------------------------------
# Training (notebook section 4)
# ----------------------------------------------------------------------
def _subsample(X, y, n):
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    if len(X) > n:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=n, replace=False)
        return X.iloc[idx], y.iloc[idx]
    return X, y


def train_logistic_regression(X, y):
    s = RandomizedSearchCV(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                           {"C": [0.001, 0.01, 0.1, 1, 10, 100]}, n_iter=6, cv=3, scoring="f1",
                           random_state=RANDOM_STATE, n_jobs=-1)
    return s.fit(X, y).best_estimator_


def train_random_forest(X, y, search_sample_size=40000):
    Xs, ys = _subsample(X, y, search_sample_size)
    s = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                           {"n_estimators": [200, 300], "max_depth": [6, 8, 10, 12],
                            "min_samples_leaf": [5, 10, 20]},
                           n_iter=8, cv=3, scoring="f1", random_state=RANDOM_STATE, n_jobs=-1).fit(Xs, ys)
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **s.best_params_).fit(X, y)


def train_xgboost(X, y, search_sample_size=50000):
    """Notebook Option A (real XGBoost). If xgboost isn't installed, falls back to
    sklearn's HistGradientBoosting with an analogous grid (the notebook's 'HGB-equivalent')."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xs, ys = _subsample(X, y, search_sample_size)
    if xgb is not None:
        grid = {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [4, 6, 8], "n_estimators": [150, 250],
                "min_child_weight": [5, 10, 20], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}
        mk = lambda **p: xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, **p)
    else:
        grid = {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [4, 6, 8], "max_iter": [150, 250],
                "min_samples_leaf": [5, 10, 20]}
        mk = lambda **p: HistGradientBoostingClassifier(random_state=RANDOM_STATE, **p)
    s = RandomizedSearchCV(mk(), grid, n_iter=10, cv=3, scoring="f1",
                           random_state=RANDOM_STATE, n_jobs=-1).fit(Xs, ys)
    return mk(**s.best_params_).fit(X, y)


def find_best_threshold(y_val, proba, metric="f1"):
    """Notebook: scan 0.05-0.95 on the VALIDATION set only."""
    best_t, best = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        p = (proba >= t).astype(int)
        score = {"f1": f1_score, "recall": recall_score}.get(metric, precision_score)(y_val, p, zero_division=0)
        if score > best:
            best, best_t = score, float(t)
    return best_t


# ----------------------------------------------------------------------
# Evaluation / prediction
# ----------------------------------------------------------------------
def predict_proba(bundle, X):
    """Probability of default. Applies the saved scaler when the model needs one."""
    X = X[bundle["feature_cols"]]          # enforce the model's own column order
    if bundle.get("scaler") is not None:
        X = pd.DataFrame(bundle["scaler"].transform(X), columns=bundle["feature_cols"], index=X.index)
    model = bundle["model"]
    if isinstance(model, LogisticRegression) and not hasattr(model, "multi_class"):
        # Model was saved with a newer scikit-learn than the one installed here (older versions
        # look for a `multi_class` attribute that newer ones no longer store). For a binary
        # logistic regression predict_proba is just sigmoid(X @ coef + intercept), so compute it directly.
        z = np.asarray(X, dtype=float) @ model.coef_.ravel() + float(model.intercept_[0])
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict_proba(X)[:, 1]


def evaluate(y_true, proba, threshold):
    """Notebook evaluate_model() plus confusion matrix and classification report."""
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "report": classification_report(y_true, pred, labels=[0, 1],
                                        target_names=[LABEL_MAP[0], LABEL_MAP[1]],
                                        zero_division=0, output_dict=True)}


def feature_importance(bundle, top=10):
    m, cols = bundle["model"], bundle["feature_cols"]
    if hasattr(m, "feature_importances_"):
        vals = m.feature_importances_
    elif hasattr(m, "coef_"):
        vals = np.abs(m.coef_[0])           # |coefficient| on scaled features
    else:
        return None
    return pd.Series(vals, index=cols).sort_values(ascending=False).head(top)


# ----------------------------------------------------------------------
# Bundles: save / load
# ----------------------------------------------------------------------
def train_and_save(train_prepared, test_prepared=None, models=("xgboost", "random_forest", "logistic_regression"),
                   log=print):
    """Run the notebook's modelling flow and save one .joblib bundle per model."""
    sep = test_prepared is not None
    if sep:
        X_tr, X_val, y_tr, y_val = split_data(train_prepared, separate_test=True)
        feats = list(X_tr.columns)
        X_te, y_te = test_prepared.reindex(columns=feats), test_prepared[TARGET]
    else:
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(train_prepared)
        feats = list(X_tr.columns)
    log(f"Split -> train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")
    X_sm, y_sm = smote_oversample_minority(X_tr, y_tr)
    scaler = StandardScaler().fit(X_sm)    # fit on SMOTE'd train only
    trainers = {"xgboost": (train_xgboost, False), "random_forest": (train_random_forest, False),
                "logistic_regression": (train_logistic_regression, True)}
    MODEL_DIR.mkdir(exist_ok=True)
    out = {}
    for key in models:
        fn, scaled = trainers[key]
        log(f"Training {MODEL_NAMES[key]} ...")
        Xfit = pd.DataFrame(scaler.transform(X_sm), columns=feats) if scaled else X_sm
        model = fn(Xfit, y_sm)
        name = MODEL_NAMES[key] + ("" if key != "xgboost" or xgb is not None else " (HGB fallback)")
        bundle = {"model": model, "model_name": name, "feature_cols": feats,
                  "scaler": scaler if scaled else None, "default_threshold": 0.5,
                  "LABEL_MAP": LABEL_MAP, "created": datetime.now().isoformat(timespec="seconds"),
                  "test_source": "uploaded test file" if sep else "held-out 20% split"}
        bundle["tuned_threshold"] = find_best_threshold(y_val, predict_proba(bundle, X_val))
        proba = predict_proba(bundle, X_te)
        bundle["test_metrics"] = {"default": evaluate(y_te, proba, 0.5),
                                  "tuned": evaluate(y_te, proba, bundle["tuned_threshold"])}
        joblib.dump(bundle, MODEL_DIR / f"{key}.joblib")
        out[key] = bundle
        m = bundle["test_metrics"]["default"]
        log(f"  saved models/{key}.joblib  tuned thr={bundle['tuned_threshold']:.2f}  "
            f"test@0.5: acc={m['accuracy']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}")
    write_metrics_json(out)
    json.dump({"feature_columns": feats, "tuned_thresholds": {k: b["tuned_threshold"] for k, b in out.items()}},
              open(MODEL_DIR / "deployment_config.json", "w"), indent=2)
    return out


def load_bundle(obj, config=None, scaler=None):
    """Accept (a) a bundle saved by this project or (b) a raw notebook model
    (final_model_*.pkl) with the optional deployment_config.json / final_scaler.pkl."""
    if isinstance(obj, dict) and "model" in obj and "feature_cols" in obj:
        return obj
    model, config = obj, config or {}
    cols = (config.get("feature_columns") or getattr(model, "feature_names_in_", None))
    if cols is None and hasattr(model, "get_booster"):
        cols = model.get_booster().feature_names
    if cols is None:
        raise ValueError("Cannot tell which features this model expects. Upload deployment_config.json "
                         "(it contains 'feature_columns') alongside the model.")
    is_lr = isinstance(model, LogisticRegression)
    if is_lr and scaler is None:
        raise ValueError("This is a Logistic Regression model trained on scaled data - "
                         "upload final_scaler.pkl as well.")
    return {"model": model, "model_name": type(model).__name__, "feature_cols": list(cols),
            "scaler": scaler if is_lr else None, "default_threshold": 0.5,
            "tuned_threshold": config.get("tuned_threshold"), "LABEL_MAP": LABEL_MAP,
            "test_metrics": None, "test_source": "n/a"}


def load_all_loans(raw_files):
    """Dashboard overview data: EVERY labelled loan from every supplied raw CSV (train + test).

    raw_files: list of (bytes, name). Files that are missing, fail validation, or have no
    LOAN_DEFAULT column are skipped. Returns (prepared_dataframe_or_None, [(name, n_rows), ...]).
    prepare_data() never drops rows, so len(result) == sum of the row counts used.
    """
    frames, used = [], []
    for raw, name in raw_files:
        if raw is None:
            continue
        df = pd.read_csv(io.BytesIO(raw))
        if validate_raw_dataset(df, True):
            continue
        frames.append(df)
        used.append((name, len(df)))
    if not frames:
        return None, used
    return prepare_data(pd.concat(frames, ignore_index=True), drop_zero_var=False), used


def _scalars(d):
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating, np.integer))}


def write_metrics_json(bundles):
    """Copy each model's stored test metrics into models/model_metrics.json so the dashboard can still
    list every model in its comparison table even when a .joblib cannot be loaded (e.g. the installed
    scikit-learn is older than the one that saved it)."""
    path = MODEL_DIR / "model_metrics.json"
    data = json.load(open(path)) if path.exists() else {}
    for key, b in bundles.items():
        data[key] = {"model_name": b["model_name"], "test_source": b.get("test_source", ""),
                     "test_metrics": {k: _scalars(v) for k, v in (b.get("test_metrics") or {}).items()}}
    json.dump(data, open(path, "w"), indent=2)
