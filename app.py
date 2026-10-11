"""Student Salary Prediction - Streamlit app."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(page_title="Student Salary Prediction", page_icon="🎓", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_salary.csv"

REQUIRED_COLUMNS = [
    "age", "gender", "education_level", "cgpa", "internship_months",
    "projects", "certifications", "skills_score", "communication_score",
    "college_tier", "placement_status", "salary"
]


def create_demo_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 27, n)
    gender = rng.choice(["Male", "Female"], n)
    education_level = rng.choice(["BTech", "BCA", "MCA", "MTech", "BSc", "MBA"], n)
    cgpa = np.clip(rng.normal(7.7, 0.9, n), 5, 10).round(2)
    internship_months = np.clip(rng.poisson(5, n), 0, 18)
    projects = np.clip(rng.poisson(3, n), 0, 10)
    certifications = np.clip(rng.poisson(2, n), 0, 8)
    skills_score = np.clip(rng.normal(70, 14, n), 30, 100).round(1)
    communication_score = np.clip(rng.normal(68, 15, n), 30, 100).round(1)
    college_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.20, 0.45, 0.35])
    placement_status = rng.choice(["Placed", "Not Placed"], n, p=[0.82, 0.18])
    tier_bonus = np.select([college_tier == "Tier 1", college_tier == "Tier 2"], [160000, 70000], default=0)
    degree_bonus = np.select([education_level == "MTech", education_level == "MBA", education_level == "MCA"], [80000, 60000, 40000], default=0)
    placement_bonus = np.where(placement_status == "Placed", 50000, -180000)
    salary = (180000 + cgpa * 85000 + internship_months * 18000 + projects * 30000
              + certifications * 12000 + skills_score * 7000 + communication_score * 3500
              + tier_bonus + degree_bonus + placement_bonus + rng.normal(0, 90000, n))
    salary = np.clip(salary, 120000, 1800000).round(-3).astype(int)
    return pd.DataFrame({
        "age": age, "gender": gender, "education_level": education_level, "cgpa": cgpa,
        "internship_months": internship_months, "projects": projects,
        "certifications": certifications, "skills_score": skills_score,
        "communication_score": communication_score, "college_tier": college_tier,
        "placement_status": placement_status, "salary": salary
    })


def load_data(uploaded_file=None):
    """Always return a DataFrame; never leave df undefined."""
    if uploaded_file is not None:
        candidate = pd.read_csv(uploaded_file)
        source = "Uploaded CSV"
    elif DATA_FILE.exists():
        try:
            candidate = pd.read_csv(DATA_FILE)
            source = f"{DATA_FILE.name}"
        except Exception:
            candidate = pd.DataFrame()
            source = ""
    else:
        candidate = pd.DataFrame()
        source = ""

    candidate.columns = candidate.columns.astype(str).str.strip()
    candidate = candidate.loc[:, ~candidate.columns.str.match(r"^Unnamed")]
    missing = [c for c in REQUIRED_COLUMNS if c not in candidate.columns]
    if candidate.empty or missing:
        candidate = create_demo_dataset()
        source = "Generated demo dataset (CSV missing, empty, or required columns missing)"
        if missing:
            st.warning("CSV mein required columns missing hain: " + ", ".join(missing) + ". Demo data use ho raha hai.")
    candidate["salary"] = pd.to_numeric(candidate["salary"], errors="coerce")
    candidate = candidate.dropna(subset=["salary"]).reset_index(drop=True)
    if candidate.empty:
        candidate = create_demo_dataset()
        source = "Generated demo dataset (salary values were invalid)"
    return candidate, source


st.title("🎓 Student Salary Prediction")
st.caption("Regression predicts estimated annual salary; classification predicts Low / Medium / High category.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Optional: upload your student_salary.csv", type=["csv"])

# IMPORTANT: df is initialized here before any display or training uses it.
df, data_source = load_data(uploaded_file)
st.success(f"Dataset loaded: {data_source}")

left, middle, right = st.columns(3)
left.metric("Rows", f"{len(df):,}")
middle.metric("Columns", len(df.columns))
right.metric("Average salary", f"₹{df['salary'].mean():,.0f}")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Missing values per column")
    st.dataframe(df.isna().sum().rename("Missing values"))

# Create salary classes from the loaded dataframe.
q1, q2 = df["salary"].quantile([1/3, 2/3]).tolist()
def salary_category(value):
    if value <= q1:
        return "Low"
    if value <= q2:
        return "Medium"
    return "High"

df = df.copy()
df["salary_category"] = df["salary"].apply(salary_category)
X = df.drop(columns=["salary", "salary_category"])
y_reg = df["salary"]
y_clf = df["salary_category"]

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

# Compatibility with old and new scikit-learn versions.
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), cat_cols),
])

st.subheader("Train and evaluate models")
if st.button("Train models", type="primary"):
    if len(df) < 10:
        st.error("Training ke liye kam se kam 10 valid rows chahiye.")
        st.stop()
    stratify_values = y_clf if y_clf.value_counts().min() >= 2 else None
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=stratify_values
    )
    reg_models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=120, random_state=42),
    }
    clf_models = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000),
    }
    reg_results, trained_reg = [], {}
    for name, estimator in reg_models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, yr_train)
        pred = pipe.predict(X_test)
        reg_results.append({"Model": name, "MAE (₹)": mean_absolute_error(yr_test, pred), "R²": r2_score(yr_test, pred)})
        trained_reg[name] = pipe
    clf_results, trained_clf = [], {}
    for name, estimator in clf_models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, yc_train)
        pred = pipe.predict(X_test)
        clf_results.append({"Model": name, "Accuracy": accuracy_score(yc_test, pred)})
        trained_clf[name] = pipe
    reg_table = pd.DataFrame(reg_results).sort_values("R²", ascending=False)
    clf_table = pd.DataFrame(clf_results).sort_values("Accuracy", ascending=False)
    best_reg_name = reg_table.iloc[0]["Model"]
    best_clf_name = clf_table.iloc[0]["Model"]
    st.subheader("Regression results")
    st.dataframe(reg_table.style.format({"MAE (₹)": "₹{:,.0f}", "R²": "{:.3f}"}), use_container_width=True)
    st.subheader("Classification results")
    st.dataframe(clf_table.style.format({"Accuracy": "{:.1%}"}), use_container_width=True)
    st.session_state["salary_reg_model"] = trained_reg[best_reg_name]
    st.session_state["salary_clf_model"] = trained_clf[best_clf_name]
    st.session_state["salary_best_reg_name"] = best_reg_name
    st.session_state["salary_best_clf_name"] = best_clf_name
    st.success(f"Training complete. Best regression: {best_reg_name}; best classification: {best_clf_name}.")

if "salary_reg_model" in st.session_state and "salary_clf_model" in st.session_state:
    st.subheader("Predict salary for a student")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=18, max_value=60, value=22)
        gender = c2.selectbox("Gender", sorted(df["gender"].dropna().astype(str).unique()))
        education = c3.selectbox("Education level", sorted(df["education_level"].dropna().astype(str).unique()))
        cgpa = c1.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        internship = c2.number_input("Internship months", min_value=0, max_value=60, value=6)
        projects = c3.number_input("Projects", min_value=0, max_value=100, value=4)
        certs = c1.number_input("Certifications", min_value=0, max_value=100, value=3)
        skills = c2.slider("Skills score", 0, 100, 80)
        communication = c3.slider("Communication score", 0, 100, 75)
        tier = c1.selectbox("College tier", sorted(df["college_tier"].dropna().astype(str).unique()))
        placement = c2.selectbox("Placement status", sorted(df["placement_status"].dropna().astype(str).unique()))
        submitted = st.form_submit_button("Predict salary")
    if submitted:
        student = pd.DataFrame([{
            "age": age, "gender": gender, "education_level": education, "cgpa": cgpa,
            "internship_months": internship, "projects": projects, "certifications": certs,
            "skills_score": skills, "communication_score": communication,
            "college_tier": tier, "placement_status": placement
        }])
        predicted_salary = st.session_state["salary_reg_model"].predict(student)[0]
        predicted_class = st.session_state["salary_clf_model"].predict(student)[0]
        p1, p2 = st.columns(2)
        p1.metric("Estimated annual salary", f"₹{predicted_salary:,.0f}")
        p2.metric("Predicted salary category", str(predicted_class))
else:
    st.info("Prediction se pehle 'Train models' button dabao.")

st.caption("Note: Demo data se nikle predictions illustrative hain; real-world accuracy ke liye verified, representative dataset use karein.")
