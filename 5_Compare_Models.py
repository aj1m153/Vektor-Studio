import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Compare Models", layout="wide")
st.title("Compare Models")

# ── Check dataset ─────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

# ── Models ───────────────────────────────────────────────
CLF_MODELS = {
    "Logistic Regression": ("sklearn.linear_model", "LogisticRegression", {"C": 1.0, "max_iter": 500}),
    "Random Forest": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100}),
    "Gradient Boosting": ("sklearn.ensemble", "GradientBoostingClassifier", {"n_estimators": 100}),
    "XGBoost": ("xgboost", "XGBClassifier", {"n_estimators": 100}),
    "LightGBM": ("lightgbm", "LGBMClassifier", {"n_estimators": 100}),
    "SVM": ("sklearn.svm", "SVC", {"C": 1.0, "kernel": "rbf"}),
    "KNN": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
}

REG_MODELS = {
    "Linear Regression": ("sklearn.linear_model", "LinearRegression", {}),
    "Ridge": ("sklearn.linear_model", "Ridge", {"alpha": 1.0}),
    "Lasso": ("sklearn.linear_model", "Lasso", {"alpha": 0.1}),
    "Random Forest": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 100}),
}

# ── Step 1 ───────────────────────────────────────────────
st.subheader("Step 1 — Configure Data")

task = st.radio("Task type", ["Classification", "Regression"], horizontal=True)
MODEL_CATALOGUE = CLF_MODELS if task == "Classification" else REG_MODELS

cols = df.columns.tolist()
target_col = st.selectbox("Target column", cols)
feature_cols = st.multiselect("Feature columns", [c for c in cols if c != target_col],
                             default=[c for c in cols if c != target_col])

if not feature_cols:
    st.stop()

seed = st.number_input("Random seed", 0, 9999, 42)
test_size = st.slider("Test size (%)", 10, 40, 20) / 100

# ── Step 2 ───────────────────────────────────────────────
st.subheader("Step 2 — Select Models")

selected_models = st.multiselect(
    "Choose models",
    list(MODEL_CATALOGUE.keys()),
    default=list(MODEL_CATALOGUE.keys())[:3]
)

if not selected_models:
    st.stop()

# ── Step 3 ───────────────────────────────────────────────
st.subheader("Step 3 — Configure Parameters")

model_configs = {}
for m in selected_models:
    _, _, defaults = MODEL_CATALOGUE[m]
    params = {}
    with st.expander(m):
        for k, v in defaults.items():
            if isinstance(v, int):
                params[k] = st.number_input(f"{m}_{k}", value=v)
            elif isinstance(v, float):
                params[k] = st.number_input(f"{m}_{k}", value=v)
    model_configs[m] = params

# ── Step 4 ───────────────────────────────────────────────
st.subheader("Step 4 — Train")

if st.button("Run Comparison"):

    X = df[feature_cols]
    y = df[target_col]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=int(seed),
        stratify=y if task == "Classification" else None
    )

    results = []
    trained_models = {}

    for model_name in selected_models:
        module_name, class_name, _ = MODEL_CATALOGUE[model_name]

        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)

            model = cls(**model_configs.get(model_name, {}))

            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            trained_models[model_name] = pipe

            if task == "Classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted")

                auc = None
                if hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X_test)
                        auc = roc_auc_score(y_test, proba, multi_class="ovr")
                    except:
                        pass

                row = {
                    "Model": model_name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                    "ROC-AUC": auc
                }

            else:
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                row = {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

            results.append(row)

        except Exception as e:
            results.append({"Model": model_name, "Error": str(e)})

    st.session_state["results"] = pd.DataFrame(results)
    st.session_state["models"] = trained_models

# ── Results ───────────────────────────────────────────────
if "results" in st.session_state:

    results_df = st.session_state["results"]

    st.subheader("Metrics Table")
    st.dataframe(results_df)

    # Download metrics
    st.download_button("Download Metrics CSV",
                       results_df.to_csv(index=False).encode(),
                       "metrics.csv")

    # ── Visual Comparison (UPDATED) ───────────────────────
    st.subheader("Visual Comparison")

    metric_cols = [c for c in results_df.columns if c != "Model"]

    chosen_metric = st.selectbox(
        "Choose metric",
        metric_cols,
        index=metric_cols.index("Precision") if "Precision" in metric_cols else 0
    )

    plot_df = results_df.dropna(subset=[chosen_metric])

    fig, ax = plt.subplots()

    vals = plot_df[chosen_metric].values
    best_idx = vals.argmax() if chosen_metric not in ["RMSE", "MAE"] else vals.argmin()

    bars = ax.bar(plot_df["Model"], vals)

    for i, bar in enumerate(bars):
        if i == best_idx:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)

    ax.set_title(f"{chosen_metric} Comparison")
    ax.tick_params(axis="x", rotation=30)

    st.pyplot(fig)

    st.success(f"Best model on {chosen_metric}: {plot_df.iloc[best_idx]['Model']}")

    # ── Export Predictions ───────────────────────────────
    st.subheader("Export Predictions")

    model_choice = st.selectbox("Select model", list(st.session_state["models"].keys()))

    if st.button("Download Predictions"):
        model = st.session_state["models"][model_choice]
        preds = model.predict(df[feature_cols])

        out = df.copy()
        out["prediction"] = preds

        st.download_button("Download CSV",
                           out.to_csv(index=False).encode(),
                           "predictions.csv")
