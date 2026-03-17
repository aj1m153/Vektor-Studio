import streamlit as st
import pandas as pd
from thefuzz import fuzz, process

st.set_page_config(page_title="Standardize", layout="wide")
st.title("Standardize Column Values")
st.markdown(
    "Merge inconsistent representations of the same value within a column "
    "(e.g. 'Angel A.' and 'Angel Andrew'). "
    "Fuzzy matching surfaces similar values automatically. All changes are queued before being applied."
)

if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
if not cat_cols:
    st.info("No text or categorical columns found in the current dataset.")
    st.stop()

col_sel = st.selectbox("Select column to standardize", cat_cols)
series = df[col_sel].dropna().astype(str)
unique_vals = sorted(series.unique().tolist())

st.markdown(f"**{len(unique_vals)}** unique values in `{col_sel}`")
with st.expander("View all unique values"):
    st.dataframe(
        series.value_counts().rename("count").reset_index().rename(columns={"index": "value"}),
        use_container_width=True
    )

st.divider()

# ── FUZZY SUGGESTIONS ─────────────────────────────────────────────────────────
st.subheader("Fuzzy Match Suggestions")
threshold = st.slider(
    "Similarity threshold (%)", 50, 100, 80,
    help="Higher values require closer matches. 80% is recommended as a starting point."
)

if st.button("Find similar values"):
    groups = []
    visited = set()
    for val in unique_vals:
        if val in visited:
            continue
        matches = process.extract(val, unique_vals, scorer=fuzz.token_sort_ratio, limit=20)
        cluster = [m[0] for m in matches if m[1] >= threshold and m[0] != val]
        if cluster:
            groups.append((val, cluster))
            visited.update(cluster)
            visited.add(val)
    st.session_state["fuzzy_groups"] = groups

if "fuzzy_groups" in st.session_state:
    groups = st.session_state["fuzzy_groups"]
    if not groups:
        st.info("No similar values found at this threshold. Try lowering it.")
    else:
        st.markdown(f"Found **{len(groups)}** potential merge group(s):")
        pending = st.session_state.setdefault("pending_replacements", {})

        for i, (canonical, similars) in enumerate(groups):
            with st.container():
                st.markdown(f"**Group {i+1}**")
                c1, c2, c3 = st.columns([2, 3, 2])
                keep_as = c1.text_input("Replace all with", value=canonical, key=f"keep_{i}")
                c2.multiselect("Values to replace", options=[canonical] + similars,
                               default=similars, key=f"sel_{i}")
                if c3.button("Queue this merge", key=f"queue_{i}"):
                    targets = st.session_state.get(f"sel_{i}", similars)
                    for t in targets:
                        pending[t] = keep_as
                    st.session_state["pending_replacements"] = pending
                    st.success(f"Queued {len(targets)} replacement(s) to '{keep_as}'")

st.divider()

# ── MANUAL REPLACEMENTS ───────────────────────────────────────────────────────
st.subheader("Manual Replacements")
c1, c2 = st.columns(2)
find_val = c1.selectbox("Find value", [""] + unique_vals, key="manual_find")
replace_val = c2.text_input("Replace with", key="manual_replace")

if st.button("Queue manual replacement") and find_val:
    pending = st.session_state.setdefault("pending_replacements", {})
    pending[find_val] = replace_val
    st.session_state["pending_replacements"] = pending
    st.success(f"Queued: '{find_val}' will be replaced with '{replace_val}'")

# ── PENDING QUEUE ─────────────────────────────────────────────────────────────
pending = st.session_state.get("pending_replacements", {})
if pending:
    st.divider()
    st.subheader("Pending Replacements")
    pq_df = pd.DataFrame(list(pending.items()), columns=["Find", "Replace With"])
    st.dataframe(pq_df, use_container_width=True)

    col_a, col_b = st.columns(2)
    if col_a.button("Apply all replacements", type="primary"):
        df[col_sel] = df[col_sel].astype(str).replace(pending)
        st.session_state["df"] = df
        st.session_state["pending_replacements"] = {}
        if "fuzzy_groups" in st.session_state:
            del st.session_state["fuzzy_groups"]
        st.success(f"Applied {len(pending)} replacement(s) to '{col_sel}' successfully.")
        st.rerun()
    if col_b.button("Clear queue"):
        st.session_state["pending_replacements"] = {}
        st.rerun()
