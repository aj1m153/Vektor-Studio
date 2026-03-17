# ML Studio

ML Studio is an end-to-end machine learning application built with Streamlit. It provides a structured pipeline for data ingestion, cleaning, standardization, model training, and evaluation — all through an interactive browser-based interface.

---

## Requirements

- Python 3.9 or higher
- pip

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aj1m153/ml_Studio.git
cd ml_Studio
```

### 2. Create and activate a virtual environment (recommended)

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
ml_Studio/
├── app.py                          # Application home page and entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── pages/
    ├── 1_Upload_Data.py            # Data ingestion
    ├── 2_Clean_Data.py             # Data cleaning and preparation
    ├── 3_Standardize.py            # Column value standardization
    ├── 4_Train_Model.py            # Model training and evaluation
    └── 5_Compare_Models.py         # Model comparison and reporting
```

---

## Pages

### 1. Upload Data
Upload datasets in CSV, Excel (.xlsx), JSON, or Parquet format. Supports delimiter selection for CSV files and sheet selection for Excel files. Displays a full column summary including data types, null counts, and unique value counts.

### 2. Clean Data
Provides five data cleaning operations:

- **Change Data Types** — Convert columns to int, float, string, bool, datetime, or category
- **Handle Missing Values** — Apply per-column strategies: drop rows, fill with mean/median/mode/constant, forward fill, or backward fill
- **Duplicate Rows** — View duplicate statistics and remove duplicates with keep-first, keep-last, or drop-all options
- **Unique Values** — Inspect value frequency distributions per column
- **Outlier Detection** — Detect outliers using IQR (Tukey) or Z-score methods, with options to remove rows or cap values (Winsorize)

Download the cleaned dataset as CSV or Excel at any point.

### 3. Standardize
Merge inconsistent representations of the same value within a column (e.g. "Angel A." and "Angel Andrew"). Supports:

- **Fuzzy match suggestions** — Automatically groups similar values based on a configurable similarity threshold
- **Manual replacements** — Select any value and specify its replacement directly
- All changes are queued before being applied, allowing review before committing

### 4. Train Model
Supports five machine learning task types:

| Task | Models Available |
|------|-----------------|
| Classification | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN, Decision Tree, Naive Bayes, Extra Trees |
| Regression | Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVR, KNN |
| Clustering | KMeans, DBSCAN, Agglomerative Clustering |
| Time Series | SARIMA, Exponential Smoothing (Holt-Winters) |
| Prophet | Facebook Prophet with seasonality and changepoint controls |

**Hyperparameter tuning options (per model):**
- Manual — Set parameters directly via the interface
- GridSearchCV — Define a parameter grid and run exhaustive cross-validated search
- Optuna — Run a configurable number of Bayesian optimization trials

**Output includes:**
- Metrics table (Accuracy, F1, ROC-AUC for classification; RMSE, MAE, R2 for regression; Silhouette score for clustering)
- Confusion matrix (classification)
- Feature importance or coefficient chart
- SHAP summary plot (where applicable)
- Downloadable trained model (.pkl)

### 5. Compare Models
Compare all trained models in a single view:

- Metrics table with best-value highlighting
- Bar chart comparison across any selected metric
- Best model identification
- Export full comparison as an HTML report
- Export predictions from the last trained model as CSV

---

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (must be public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** and select your repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

For Prophet on Streamlit Cloud, create a `packages.txt` file in the root with:

```
libpython3-dev
```

---

## Notes

- All data is stored in Streamlit session state and is cleared when the browser session ends. No data is persisted to disk.
- Prophet may require additional system-level dependencies on certain platforms. Refer to the [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) for details.
- SHAP values are computed automatically for tree-based and linear models. For other model types, SHAP may not be available.
- For large datasets (over 100,000 rows), model training and SHAP computation may take longer. Consider sampling the data before training if performance is a concern.

---

## License

This project is provided as-is for educational and internal use.
