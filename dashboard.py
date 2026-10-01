"""
dashboard.py - the ENTIRE dashboard in one file (Streamlit).

Replaces the old three-file HTML dashboard (dashboard_data_prep.py + build_dashboard.py +
dashboard_template.html): the segment explorer is computed live from the prepared data, and the
model evaluation / prediction / feature-importance views sit next to it.

Run via `python main.py`, or directly:  streamlit run dashboard.py
"""
import io
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P

st.set_page_config(page_title="Vehicle Loan Default", layout="wide")
BAND_ORDER = {
    "age_band": ["18-25", "26-35", "36-45", "46-55", "56+"],
    "ltv_band": ["<=60", "61-75", "76-85", "86+"],
    "asset_cost_band": ["<=50k", "50k-70k", "70k-90k", "90k+"],
    "employment_label": ["Salaried", "Self Employed"],
    "risk_label": ["Low", "No History/Not Scored", "Medium", "High"],
}
FILTERS = [("employment_label", "Employment Type"), ("age_band", "Age Band"), ("ltv_band", "LTV Band"),
           ("asset_cost_band", "Asset Cost Band"), ("risk_label", "Credit Risk Grade")]
CHARTS = [("risk_label", "Default Rate by Credit Risk Grade", "#D62828"),
          ("age_band", "Default Rate by Age Band", "#2E86AB"),
          ("employment_label", "Default Rate by Employment Type", "#F77F00"),
          ("ltv_band", "Default Rate by LTV Band", "#6A4C93"),
          ("asset_cost_band", "Default Rate by Asset Cost Band", "#2A9D8F")]


# ---------------------------------------------------------------- helpers
def first_csv(folder):
    files = sorted(folder.glob("*.csv"))
    return files[0] if files else None


def pick_source(upload, folder):
    """Uploaded file wins; otherwise the first CSV in the project's data folder."""
    if upload is not None:
        return upload.getvalue(), upload.name
    f = first_csv(folder)
    return (f.read_bytes(), f.name) if f else (None, None)


@st.cache_data(show_spinner="Preparing data (same steps as the notebook) ...")
def load_prepared(raw, drop_zero_var, need_target):
    df = pd.read_csv(io.BytesIO(raw))
    problems = P.validate_raw_dataset(df, need_target)
    return (None, problems) if problems else (P.prepare_data(df, drop_zero_var), [])


@st.cache_data(show_spinner="Preparing all loans (train + test) ...")
def load_overview(train_raw, train_name, test_raw, test_name):
    return P.load_all_loans([(train_raw, train_name), (test_raw, test_name)])


@st.cache_resource(show_spinner="Loading model ...")
def load_model_cached(model_bytes, cfg_bytes, scaler_bytes):
    obj = joblib.load(io.BytesIO(model_bytes))
    cfg = json.loads(cfg_bytes) if cfg_bytes else None
    scaler = joblib.load(io.BytesIO(scaler_bytes)) if scaler_bytes else None
    return P.load_bundle(obj, cfg, scaler)


@st.cache_data(show_spinner="Scoring ...")
def score(_bundle, key, X):
    return P.predict_proba(_bundle, X)


def add_bands(d):
    """Same bands as the old dashboard_data_prep.py. Rows that fall outside a band are kept
    (labelled "Other") so that no loan ever disappears from the dashboard."""
    d = d.copy()
    cuts = {
        "age_band": pd.cut(d["AGE"], [0, 25, 35, 45, 55, 1000], labels=BAND_ORDER["age_band"]),
        "ltv_band": pd.cut(d["LTV"], [-np.inf, 60, 75, 85, np.inf], labels=BAND_ORDER["ltv_band"]),
        "asset_cost_band": pd.cut(d["ASSET_COST"], [-np.inf, 50000, 70000, 90000, np.inf],
                                  labels=BAND_ORDER["asset_cost_band"]),
        "employment_label": d["EMPLOYMENT_TYPE"].map({1.0: "Salaried", 0.0: "Self Employed"}),
        "risk_label": d["CREDIT_RISK_CATEGORY"],
    }
    for col, ser in cuts.items():
        d[col] = ser.astype(object).where(ser.notna(), "Other")
    return d


def order_for(d, col):
    """Display order for a dimension: the known order first, then any unexpected value."""
    present = set(d[col].unique())
    return [v for v in BAND_ORDER[col] if v in present] + sorted(present - set(BAND_ORDER[col]))


def rate_chart(d, col, color):
    r = d.groupby(col)[P.TARGET].mean().reindex(order_for(d, col))
    fig, ax = plt.subplots(figsize=(5.2, 3))
    bars = ax.bar(r.index.astype(str), r.values * 100, color=color)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in r.values * 100], fontsize=8, padding=2)
    ax.set_ylabel("Default rate (%)")
    ax.set_ylim(0, max(r.max() * 100 * 1.2, 1))
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def rate_style(v):
    """Same thresholds as the HTML template's badges."""
    if v > 0.28:
        return "background-color: rgba(230,57,70,0.18); color: #E63946; font-weight: 600"
    if v > 0.20:
        return "background-color: rgba(247,127,0,0.18); color: #F77F00; font-weight: 600"
    return "background-color: rgba(46,134,171,0.18); color: #2E86AB; font-weight: 600"


def reset_filters():
    for col, _ in FILTERS:
        st.session_state[f"f_{col}"] = "All"


def model_records(paths):
    """Stored test metrics for every saved model. If a .joblib cannot be loaded (usually because the
    installed scikit-learn is older than the one that saved it) fall back to models/model_metrics.json
    so the model still appears, and report the failure instead of silently skipping it."""
    jpath = P.MODEL_DIR / "model_metrics.json"
    stored = json.loads(jpath.read_text()) if jpath.exists() else {}
    recs, failed = [], []
    for p in paths:
        rec = None
        try:
            b = load_model_cached(p.read_bytes(), None, None)
            rec = {"name": b["model_name"], "tm": b.get("test_metrics") or {}, "src": b.get("test_source", "")}
        except Exception as e:
            failed.append((p.stem, f"{type(e).__name__}: {str(e)[:120]}"))
            j = stored.get(p.stem)
            if j:
                rec = {"name": j["model_name"], "tm": j["test_metrics"], "src": j.get("test_source", "")}
        if rec:
            recs.append(rec)
    return recs, failed


def stored_model_metrics(paths):
    """One row per saved model (validation-tuned threshold, falling back to 0.50)."""
    rows = []
    recs, failed = model_records(paths)
    for r in recs:
        m = r["tm"].get("tuned") or r["tm"].get("default")
        if m:
            rows.append({"model": r["name"], "threshold": m["threshold"], "accuracy": m["accuracy"],
                         "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "auc": m["auc"]})
    return rows, failed


def confusion_fig(cm):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(cm, cmap="Blues")
    names = [P.LABEL_MAP[0], P.LABEL_MAP[1]]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("LOAN DEFAULT MODEL")
sb.header("Data")
full_up = sb.file_uploader("Upload FULL dataset (all 233,154 loans) - used by the Overview", type="csv", key="full")
train_up = sb.file_uploader("Upload Training Dataset (raw train.csv)", type="csv", key="train")
test_up = sb.file_uploader("Upload Testing Dataset (raw, labelled)", type="csv", key="test")
sb.header("Model")
saved = sorted(P.MODEL_DIR.glob("*.joblib"))
saved_choice = sb.selectbox("Saved model (models/ folder)", saved, format_func=lambda p: p.stem) if saved else None
model_up = sb.file_uploader("...or upload a saved model (.joblib / .pkl)", type=["joblib", "pkl"], key="model")
with sb.expander("Extra files for raw notebook models"):
    cfg_up = st.file_uploader("deployment_config.json", type="json", key="cfg")
    scaler_up = st.file_uploader("final_scaler.pkl (Logistic Regression only)", type="pkl", key="scaler")

if "results" not in st.session_state:
    st.session_state["results"] = {}

# ---------------------------------------------------------------- load data
train_raw, train_name = pick_source(train_up, P.TRAIN_DIR)
test_raw, test_name = pick_source(test_up, P.TEST_DIR)
errors = []
train_df = test_df = None
if train_raw:
    train_df, probs = load_prepared(train_raw, True, True)
    errors += [f"Training file '{train_name}': {p}" for p in probs]
if test_raw:
    test_df, probs = load_prepared(test_raw, False, False)
    errors += [f"Testing file '{test_name}': {p}" for p in probs]

# Overview uses EVERY labelled loan: training file + testing file combined.
full_raw, full_name = pick_source(full_up, P.FULL_DIR)
if full_raw:   # the complete dataset on its own (so the train/test samples are not double counted)
    overview_raw, overview_used = load_overview(full_raw, full_name, None, None)
else:          # fallback: the train + test samples
    overview_raw, overview_used = load_overview(train_raw, train_name, test_raw, test_name)

# ---------------------------------------------------------------- load model
bundle = None
try:
    if model_up is not None:
        bundle = load_model_cached(model_up.getvalue(), cfg_up.getvalue() if cfg_up else None,
                                   scaler_up.getvalue() if scaler_up else None)
        model_label = model_up.name
    elif saved_choice is not None:
        bundle = load_model_cached(saved_choice.read_bytes(), None, None)
        model_label = saved_choice.stem
except Exception as e:  # unreadable / incompatible file
    errors.append(f"Model could not be loaded: {e}")

# Evaluation set: uploaded/dropped-in test file, else the notebook's held-out 20% split.
X_eval = y_eval = None
eval_source = None
if test_df is not None:
    X_eval = test_df.drop(columns=[c for c in [P.TARGET, "CREDIT_RISK_CATEGORY"] if c in test_df.columns])
    y_eval = test_df[P.TARGET] if P.TARGET in test_df.columns else None
    eval_source = f"uploaded testing file ({test_name})"
elif train_df is not None:
    _, _, X_eval, _, _, y_eval = P.split_data(train_df)
    eval_source = "held-out 20% test split of the training file (notebook split, random_state=42)"

st.markdown(
    f"""<div style="background:#1a2332;padding:22px 28px;border-radius:10px;margin-bottom:18px;">
    <div style="color:#fff;font-size:22px;font-weight:700;">🚗 Vehicle Loan Default Risk Dashboard</div>
    <div style="color:#aab4c4;font-size:13px;margin-top:4px;">L&amp;T Financial Services — 233,154 loans |
    Honest, leakage-free evaluation (test set never balanced or seen during training)</div></div>""",
    unsafe_allow_html=True)
for e in errors:
    st.error(e)

tabs = st.tabs(["Overview", "Model & compatibility", "Evaluation", "Prediction", "Feature importance"])

# ---------------------------------------------------------------- Overview (HTML template layout)
with tabs[0]:
    if overview_raw is None:
        st.info("Upload a labelled training and/or testing dataset in the sidebar "
                "(or place the CSVs in data/training/ and data/testing/).")
    else:
        if not full_raw:
            st.warning(f"Overview is showing only the {len(overview_raw):,} loans in your train + test files. "
                       "To show all 233,154 loans, put the full labelled CSV in data/full/ "
                       "or upload it in the sidebar.")
        all_loans = add_bands(overview_raw)
        model_rows, load_failures = stored_model_metrics(saved)
        best = max(model_rows, key=lambda r: r["auc"]) if model_rows else None

        for name, err in load_failures:
            st.warning(f"Could not load models/{name}.joblib ({err}). Its stored metrics are still shown, but it "
                       "cannot be used for scoring. Fix: pip install -U scikit-learn (the models were saved with "
                       "scikit-learn 1.8).")

        # KPI cards
        k = st.columns(4)
        for c, label, value, help_ in [
            (k[0], "Total Loans", f"{len(all_loans):,}", " + ".join(f"{n:,} ({name})" for name, n in overview_used)),
            (k[1], "Overall Default Rate", f"{all_loans[P.TARGET].mean():.1%}", None),
            (k[2], "Best Model (AUC)", f"{best['auc']:.3f}" if best else "-", best["model"] if best else None),
            (k[3], "Best Model F1-score", f"{best['f1']:.3f}" if best else "-", best["model"] if best else None),
        ]:
            with c.container(border=True):
                st.metric(label, value, help=help_)

        # Filters
        with st.container(border=True):
            st.markdown("**🔍 Filter Segments**")
            fc = st.columns(5)
            d = all_loans
            for c, (col, label) in zip(fc, FILTERS):
                choice = c.selectbox(label, ["All"] + order_for(all_loans, col), key=f"f_{col}")
                if choice != "All":
                    d = d[d[col] == choice]
            st.button("Reset Filters", on_click=reset_filters, type="primary")

        if d.empty:
            st.warning("No loans match the current filters.")
        else:
            st.caption(f"{len(d):,} of {len(all_loans):,} loans match the current filters "
                       f"(default rate {d[P.TARGET].mean():.1%}).")

            # charts: 3 on the first row, 2 on the second
            for row in (CHARTS[:3], CHARTS[3:]):
                for c, (col, title, color) in zip(st.columns(3), row):
                    with c.container(border=True):
                        st.markdown(f"**{title}**")
                        st.pyplot(rate_chart(d, col, color))

            # Segment table: ALL segments (no minimum size, no top-N cut-off)
            seg = (d.groupby([c for c, _ in FILTERS])[P.TARGET].agg(["count", "mean"]).reset_index()
                   .sort_values(["count", "mean"], ascending=False))
            seg.columns = ["Employment", "Age", "LTV", "Asset Cost", "Risk Grade", "Loans", "Default Rate"]
            with st.container(border=True):
                st.markdown("**Filtered Segments (matching current filters)**")
                sty = seg.style.format({"Loans": "{:,}", "Default Rate": "{:.1%}"})
                sty = (sty.map if hasattr(sty, "map") else sty.applymap)(rate_style, subset=["Default Rate"])
                st.dataframe(sty, use_container_width=True, hide_index=True, height=320)
                st.caption(f"All {len(seg):,} segments shown, covering {int(seg['Loans'].sum()):,} loans "
                           f"(= every loan matching the filters). Default rate: "
                           ":blue[<= 20%]  :orange[20-28%]  :red[> 28%]")

# ---------------------------------------------------------------- Model & compatibility
with tabs[1]:
    if bundle is None:
        st.info("Run `python main.py` to train and save models, or upload a saved model in the sidebar.")
    else:
        c = st.columns(4)
        c[0].metric("Model", bundle["model_name"])
        c[1].metric("Expected features", len(bundle["feature_cols"]))
        c[2].metric("Scaler required", "Yes" if bundle.get("scaler") is not None else "No")
        c[3].metric("Tuned threshold", f"{bundle['tuned_threshold']:.2f}" if bundle.get("tuned_threshold") else "n/a")
        st.write(f"**Classes:** {bundle.get('LABEL_MAP', P.LABEL_MAP)}")
        if X_eval is None:
            st.warning("No data loaded yet, so compatibility can't be checked.")
        else:
            ok, missing, extra, msgs = P.validate_model_features(bundle, list(X_eval.columns))
            st.write(f"Model expects **{len(bundle['feature_cols'])}** features; the {eval_source} provides "
                     f"**{X_eval.shape[1]}** columns after preparation.")
            if ok:
                st.success("Compatible. Columns are re-ordered to the model's own feature order before scoring."
                           + (f" {len(extra)} extra column(s) are ignored: {extra}" if extra else ""))
            else:
                st.error("Incompatible.\n\n" + "\n\n".join(msgs) +
                         "\n\nUpload a model trained on this feature set, or check that the file is the raw dataset.")
        with st.expander("Feature list (model order)"):
            st.write(bundle["feature_cols"])

# helper state for the remaining tabs
compatible = False
if bundle is not None and X_eval is not None:
    compatible = P.validate_model_features(bundle, list(X_eval.columns))[0]

thr = None
if bundle is not None:
    tuned = bundle.get("tuned_threshold")
    opts = ["Default (0.50)"] + ([f"Tuned on validation ({tuned:.2f})"] if tuned else []) + ["Custom"]
    choice = sb.radio("Decision threshold", opts)
    thr = 0.5 if choice.startswith("Default") else (tuned if choice.startswith("Tuned") else
                                                    sb.slider("Custom threshold", 0.05, 0.95, 0.5, 0.01))

# ---------------------------------------------------------------- Evaluation
with tabs[2]:
    if not compatible:
        st.info("Load a compatible model and data first (see 'Model & compatibility').")
    elif y_eval is None:
        st.warning(f"The testing file has no '{P.TARGET}' column, so metrics can't be computed. "
                   "Predictions are still available in the Prediction tab.")
    else:
        st.caption(f"Evaluating on: {eval_source}")
        if st.button("Run Model", type="primary"):
            proba = score(bundle, f"{model_label}|{eval_source}", X_eval)
            res = P.evaluate(y_eval, proba, thr)
            res.update(model=bundle["model_name"], source=eval_source)
            st.session_state["results"][f"{model_label} @ {thr:.2f} on {eval_source}"] = res
            st.session_state["last"] = res
        res = st.session_state.get("last")
        if res:
            m = st.columns(5)
            for c, k in zip(m, ["accuracy", "precision", "recall", "f1", "auc"]):
                c.metric(k.upper() if k == "auc" else k.capitalize(), f"{res[k]:.3f}")
            st.caption(f"{res['model']} at threshold {res['threshold']:.2f}")
            a, b = st.columns(2)
            a.pyplot(confusion_fig(res["confusion_matrix"]))
            b.dataframe(pd.DataFrame(res["report"]).T.round(3))

# ---------------------------------------------------------------- Prediction
with tabs[3]:
    if not compatible:
        st.info("Load a compatible model and data first.")
    else:
        proba = score(bundle, f"{model_label}|{eval_source}", X_eval)
        i = st.number_input("Loan row (position in the evaluation data)", 0, len(X_eval) - 1, 0)
        p = float(proba[int(i)])
        label = P.LABEL_MAP[int(p >= thr)]
        c = st.columns(3)
        c[0].metric("Predicted", label)
        c[1].metric("Probability of default", f"{p:.1%}")
        c[2].metric("Threshold used", f"{thr:.2f}")
        st.write({P.LABEL_MAP[0]: f"{1 - p:.1%}", P.LABEL_MAP[1]: f"{p:.1%}"})
        if y_eval is not None:
            st.write(f"Actual outcome: **{P.LABEL_MAP[int(y_eval.iloc[int(i)])]}**")
        out = X_eval.copy()
        out["default_probability"] = proba
        out["prediction"] = [P.LABEL_MAP[int(v >= thr)] for v in proba]
        st.download_button("Download predictions for all rows (CSV)", out.to_csv(index=False), "predictions.csv")

# ---------------------------------------------------------------- Feature importance
with tabs[4]:
    if bundle is None:
        st.info("Load a model first.")
    else:
        imp = P.feature_importance(bundle, top=10)
        if imp is None:
            st.info("This model type doesn't expose feature importances.")
        else:
            kind = "|coefficient| (scaled features)" if hasattr(bundle["model"], "coef_") else "importance"
            fig, ax = plt.subplots(figsize=(7, 4))
            imp.sort_values().plot(kind="barh", ax=ax, color="#F77F00")
            ax.set_title(f"Top 10 features - {bundle['model_name']} ({kind})")
            fig.tight_layout()
            st.pyplot(fig)
            st.dataframe(imp.rename("importance").to_frame().round(4))
