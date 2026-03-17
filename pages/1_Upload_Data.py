import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload Data", layout="wide")
st.title("Upload Data")
st.markdown("Upload your dataset to get started. Supported formats: CSV, Excel, JSON, and Parquet.")

uploaded = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls", "json", "parquet"],
    help="Maximum recommended file size: 200 MB"
)

if uploaded:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()

    with st.spinner("Reading file..."):
        try:
            if ext == "csv":
                sep = st.selectbox("CSV delimiter", [",", ";", "\\t", "|"], index=0)
                sep = "\t" if sep == "\\t" else sep
                df = pd.read_csv(uploaded, sep=sep)
            elif ext in ("xlsx", "xls"):
                xl = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Select sheet", xl.sheet_names)
                df = xl.parse(sheet)
            elif ext == "json":
                df = pd.read_json(uploaded)
            elif ext == "parquet":
                df = pd.read_parquet(uploaded)
            else:
                st.error("Unsupported format.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    st.session_state["df"] = df.copy()
    st.session_state["df_original"] = df.copy()
    st.session_state["filename"] = uploaded.name

    st.success(f"Loaded {uploaded.name} — {df.shape[0]:,} rows x {df.shape[1]} columns")

    st.subheader("Preview")
    n = st.slider("Rows to preview", 5, 100, 10)
    st.dataframe(df.head(n), use_container_width=True)

    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.subheader("Column Information")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non-null": df.notnull().sum(),
        "null": df.isnull().sum(),
        "null %": (df.isnull().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    st.dataframe(info, use_container_width=True)

elif "df" in st.session_state:
    st.info(f"Dataset already loaded: {st.session_state.get('filename', 'file')}. Use the sidebar to continue.")
    st.dataframe(st.session_state["df"].head(10), use_container_width=True)
else:
    st.info("Upload a file above to begin.")
