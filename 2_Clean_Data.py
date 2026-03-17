import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(page_title="Clean Data", layout="wide")
st.title("Clean Data")

if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

def apply_df(new_df):
    st.session_state["df"] = new_df

# ── 1. DATA TYPES ─────────────────────────────────────────────────────────────
with st.expander("1. Change Data Types", expanded=False):
    st.markdown("Assign the correct data type to each column.")
    type_options = ["(keep current)", "int64", "float64", "str", "bool", "datetime64[ns]", "category"]
    cols = df.columns.tolist()
    changes = {}
    grid = st.columns(3)
    for i, col in enumerate(cols):
        current = str(df[col].dtype)
        new_type = grid[i % 3].selectbox(
            f"{col} — current: {current}",
            type_options, key=f"dtype_{col}"
        )
        if new_type != "(keep current)":
            changes[col] = new_type

    if st.button("Apply type changes", key="apply_types"):
        for col, t in changes.items():
            try:
                if t == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif t == "bool":
                    df[col] = df[col].astype(bool)
                else:
                    df[col] = df[col].astype(t)
            except Exception as e:
                st.warning(f"Could not convert {col} to {t}: {e}")
        apply_df(df)
        st.success("Data types updated successfully.")
        st.rerun()

# ── 2. MISSING VALUES ─────────────────────────────────────────────────────────
with st.expander("2. Handle Missing Values", expanded=False):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(
            pd.DataFrame({"Missing": missing, "% Missing": (missing / len(df) * 100).round(2)}),
            use_container_width=True
        )

        st.markdown("**Select a strategy per column:**")
        strategy_options = ["(skip)", "Drop rows", "Fill with mean", "Fill with median",
                            "Fill with mode", "Fill with constant", "Forward fill", "Backward fill"]
        mv_changes = {}
        for col in missing.index:
            c1, c2 = st.columns([2, 2])
            strat = c1.selectbox(f"{col}", strategy_options, key=f"mv_{col}")
            const_val = ""
            if strat == "Fill with constant":
                const_val = c2.text_input("Constant value", key=f"mv_const_{col}")
            mv_changes[col] = (strat, const_val)

        if st.button("Apply missing value strategies", key="apply_mv"):
            for col, (strat, const) in mv_changes.items():
                if strat == "(skip)":
                    continue
                elif strat == "Drop rows":
                    df = df[df[col].notna()]
                elif strat == "Fill with mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strat == "Fill with median":
                    df[col] = df[col].fillna(df[col].median())
                elif strat == "Fill with mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strat == "Fill with constant":
                    df[col] = df[col].fillna(const)
                elif strat == "Forward fill":
                    df[col] = df[col].ffill()
                elif strat == "Backward fill":
                    df[col] = df[col].bfill()
            apply_df(df)
            st.success("Missing values handled successfully.")
            st.rerun()

# ── 3. DUPLICATES ─────────────────────────────────────────────────────────────
with st.expander("3. Duplicate Rows", expanded=False):
    n_dup = df.duplicated().sum()
    pct = round(n_dup / len(df) * 100, 2)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Duplicate Rows", f"{n_dup:,}")
    col3.metric("Percentage Duplicates", f"{pct}%")

    if n_dup > 0:
        st.dataframe(df[df.duplicated(keep=False)].head(50), use_container_width=True)
        keep = st.radio("Which duplicate to keep?", ["first", "last", "Drop all"], horizontal=True)
        if st.button("Remove duplicates"):
            keep_val = False if keep == "Drop all" else keep
            df = df.drop_duplicates(keep=keep_val)
            apply_df(df)
            st.success(f"Duplicates removed. Remaining rows: {len(df):,}")
            st.rerun()
    else:
        st.success("No duplicate rows found.")

# ── 4. UNIQUE VALUES ──────────────────────────────────────────────────────────
with st.expander("4. Unique Values per Column", expanded=False):
    col_sel = st.selectbox("Select column", df.columns.tolist(), key="uniq_col")
    vc = df[col_sel].value_counts(dropna=False)
    st.markdown(f"**{df[col_sel].nunique()}** unique values in `{col_sel}`")
    st.dataframe(vc.rename("count").reset_index(), use_container_width=True)

# ── 5. OUTLIER DETECTION ──────────────────────────────────────────────────────
with st.expander("5. Outlier Detection", expanded=False):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No numeric columns found.")
    else:
        method = st.radio("Detection method", ["IQR (Tukey)", "Z-score (3 sigma)"], horizontal=True)
        col_sel_out = st.selectbox("Column to inspect", num_cols, key="out_col")
        series = df[col_sel_out].dropna()

        if method == "IQR (Tukey)":
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_mask = (df[col_sel_out] < lo) | (df[col_sel_out] > hi)
            label = f"IQR bounds: [{lo:.2f}, {hi:.2f}]"
        else:
            mu, sigma = series.mean(), series.std()
            lo, hi = mu - 3 * sigma, mu + 3 * sigma
            outlier_mask = (df[col_sel_out] < lo) | (df[col_sel_out] > hi)
            label = f"Z-score bounds: [{lo:.2f}, {hi:.2f}]"

        n_out = outlier_mask.sum()
        st.markdown(f"**{n_out}** outliers detected — {label}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].boxplot(series, vert=False)
        axes[0].set_title(f"Boxplot — {col_sel_out}")
        axes[1].hist(series, bins=40, edgecolor="none", alpha=0.8)
        axes[1].axvline(lo, color="red", linestyle="--", label="bounds")
        axes[1].axvline(hi, color="red", linestyle="--")
        axes[1].legend()
        axes[1].set_title("Distribution")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if n_out > 0:
            action = st.selectbox(
                "Action on outliers",
                ["(do nothing)", "Remove outlier rows", "Cap to bounds (Winsorize)"],
                key="out_action"
            )
            if st.button("Apply outlier action") and action != "(do nothing)":
                if action == "Remove outlier rows":
                    df = df[~outlier_mask]
                elif action == "Cap to bounds (Winsorize)":
                    df[col_sel_out] = df[col_sel_out].clip(lower=lo, upper=hi)
                apply_df(df)
                st.success(f"Outlier action applied to {col_sel_out}.")
                st.rerun()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Current Dataset")
st.caption(f"{st.session_state['df'].shape[0]:,} rows x {st.session_state['df'].shape[1]} columns")
st.dataframe(st.session_state["df"].head(20), use_container_width=True)

if st.button("Reset to original data"):
    st.session_state["df"] = st.session_state["df_original"].copy()
    st.success("Dataset reset to original.")
    st.rerun()

col_dl1, col_dl2 = st.columns(2)
csv_bytes = st.session_state["df"].to_csv(index=False).encode()
col_dl1.download_button("Download cleaned CSV", csv_bytes, "cleaned_data.csv", "text/csv")
excel_buf = __import__("io").BytesIO()
st.session_state["df"].to_excel(excel_buf, index=False)
col_dl2.download_button("Download cleaned Excel", excel_buf.getvalue(), "cleaned_data.xlsx")
