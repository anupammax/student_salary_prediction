import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(page_title="Student Salary Prediction", page_icon="🎓", layout="wide")

RANDOM_STATE = 42
DATA_FILE = "student_salary.csv"



def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "salary" not in df.columns:
            st.error("CSV mein 'salary' column nahi mila. Please correct student_salary.csv upload karein.")
            st.stop()
        return df
    st.info("student_salary.csv nahi mili, isliye demo dataset use ho raha hai. Real data ke liye CSV project folder mein rakhein.")
    return create_demo_dataset()

def make_one_hot_encoder():
    # Compatibility with both newer and older scikit-learn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

@st.cache_resource
def train_models(df):
    df = df.copy()
    df = df.dropna(subset=["salary"])
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    if len(df) < 30:
        raise ValueError("Model training ke liye kam se kam 30 rows honi chahiye.")
    if "salary_category" in df.columns:
        df = df.drop(columns=["salary_category"])
    X = df.drop(columns=["salary"])
    y = pd.to_numeric(df["salary"], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if X.empty:
        raise ValueError("Salary column mein valid numeric values nahi hain.")
    q1, q2 = y.quantile(1/3), y.quantile(2/3)
    categories = pd.cut(y, bins=[-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"], include_lowest=True)
    # Ensure stratification works with small or unusual datasets
    stratify = categories if categories.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y, categories, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder())
        ]), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers)
    reg = Pipeline([("preprocessor", preprocessor), ("model", RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_split=4, random_state=RANDOM_STATE, n_jobs=-1
    ))])
    clf = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(
        n_estimators=250, max_depth=14, min_samples_split=4, random_state=RANDOM_STATE, n_jobs=-1
    ))])
    reg.fit(X_train, y_train)
    clf.fit(X_train, c_train.astype(str))
    reg_pred = reg.predict(X_test)
    clf_pred = clf.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, reg_pred),
        "r2": r2_score(y_test, reg_pred),
        "accuracy": accuracy_score(c_test.astype(str), clf_pred)
    }
    return reg, clf, X, y, q1, q2, metrics

st.title("🎓 Student Salary Prediction")
st.caption("Machine Learning project — Regression se estimated annual salary aur Classification se salary category.")

df = load_data()
with st.expander("Dataset preview", expanded=False):
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df.head(10), use_container_width=True)

try:
    reg_model, clf_model, X, y, q1, q2, metrics = train_models(df)
except Exception as exc:
    st.error(f"Model train nahi ho saka: {exc}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Test MAE", f"₹{metrics['mae']:,.0f}")
m2.metric("Test R²", f"{metrics['r2']:.3f}")
m3.metric("Category accuracy", f"{metrics['accuracy']*100:.1f}%")
st.caption("Demo dataset use hone par metrics sirf synthetic demo data ko reflect karte hain, real-world placement guarantee nahi.")

st.divider()
st.subheader("🔮 Student details enter karein")
defaults = {
    "age": 22, "gender": "Male", "education_level": "BTech", "cgpa": 8.2,
    "internship_months": 6, "projects": 4, "certifications": 3,
    "skills_score": 82.0, "communication_score": 78.0,
    "college_tier": "Tier 2", "placement_status": "Placed"
}
input_values = {}
left, right = st.columns(2)
for idx, col in enumerate(X.columns):
    container = left if idx % 2 == 0 else right
    series = X[col]
    label = col.replace("_", " ").title()
    with container:
        if pd.api.types.is_numeric_dtype(series):
            if col == "age":
                value = int(defaults.get(col, series.median()))
                input_values[col] = st.number_input(label, min_value=16, max_value=70, value=max(16, min(70, value)), step=1)
            elif col == "cgpa":
                input_values[col] = st.number_input(label, min_value=0.0, max_value=10.0, value=float(defaults.get(col, series.median())), step=0.1)
            else:
                maximum = max(100.0, float(series.max()))
                value = float(defaults.get(col, series.median()))
                input_values[col] = st.number_input(label, min_value=0.0, max_value=maximum, value=max(0.0, min(maximum, value)), step=1.0)
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            default = str(defaults.get(col, options[0]))
            selected = default if default in options else options[0]
            input_values[col] = st.selectbox(label, options, index=options.index(selected))

if st.button("Predict Salary", type="primary", use_container_width=True):
    student = pd.DataFrame([input_values], columns=X.columns)
    predicted_salary = float(reg_model.predict(student)[0])
    predicted_category = str(clf_model.predict(student)[0])
    st.subheader("Prediction result")
    r1, r2 = st.columns(2)
    r1.metric("Estimated annual salary", f"₹{predicted_salary:,.0f}")
    r2.metric("Predicted salary category", predicted_category)
    st.info(f"Training data ke salary thresholds: Low ≤ ₹{q1:,.0f}; Medium ≤ ₹{q2:,.0f}; High > ₹{q2:,.0f}. Category classifier alag model hai, isliye category aur numeric estimate zaroori nahi ki threshold se exactly match karein.")
    st.warning("Ye educational ML estimate hai, actual job offer ya guaranteed salary nahi.")

st.divider()
st.subheader("About this project")
st.write("App student profile features se annual salary estimate karta hai aur salary ko Low, Medium, High categories mein classify karta hai. Accurate results ke liye representative, clean, real placement data use karein.")
pd.qcut(
    df["Salary"],
    q=3,
    labels=False,
    duplicates="drop"
)
