import streamlit as st

st.set_page_config(
    page_title="ML Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ML Studio")
st.markdown("""
**ML Studio** is an end-to-end machine learning workspace for data preparation, model training, and evaluation.

Use the sidebar to navigate through the pipeline:

| Step | Page | Description |
|------|------|-------------|
| 1 | **Upload Data** | Load CSV, Excel, JSON, or Parquet files |
| 2 | **Clean Data** | Handle missing values, duplicates, data types, and outliers |
| 3 | **Standardize** | Merge similar column values using fuzzy matching |
| 4 | **Train Model** | Classification, Regression, Clustering, Time Series, Prophet |
| 5 | **Compare Models** | Side-by-side model evaluation and export |

---
Get started by uploading your dataset on the **Upload Data** page.
""")

if "df" not in st.session_state:
    st.info("Navigate to the Upload Data page to load your dataset.")
else:
    df = st.session_state["df"]
    st.success(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
