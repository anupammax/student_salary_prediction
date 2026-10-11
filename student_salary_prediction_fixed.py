# ============================================================
# STUDENT SALARY PREDICTION
# REGRESSION + CLASSIFICATION MACHINE LEARNING PROJECT
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "student_salary.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
RANDOM_STATE = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. CREATE DEMO DATASET
# ============================================================

def create_demo_dataset(n=1200):
    np.random.seed(RANDOM_STATE)

    age = np.random.randint(20, 27, n)

    gender = np.random.choice(["Male", "Female"], n)

    education_level = np.random.choice(
        ["BTech", "BCA", "MCA", "MTech", "BSc", "MBA"], n
    )

    cgpa = np.clip(
        np.random.normal(7.7, 0.9, n), 5, 10
    ).round(2)

    internship_months = np.clip(
        np.random.poisson(5, n), 0, 18
    )

    projects = np.clip(
        np.random.poisson(3, n), 0, 10
    )

    certifications = np.clip(
        np.random.poisson(2, n), 0, 8
    )

    skills_score = np.clip(
        np.random.normal(70, 14, n), 30, 100
    ).round(1)

    communication_score = np.clip(
        np.random.normal(68, 15, n), 30, 100
    ).round(1)

    college_tier = np.random.choice(
        ["Tier 1", "Tier 2", "Tier 3"],
        n,
        p=[0.20, 0.45, 0.35]
    )

    placement_status = np.random.choice(
        ["Placed", "Not Placed"],
        n,
        p=[0.82, 0.18]
    )

    tier_bonus = np.select(
        [
            college_tier == "Tier 1",
            college_tier == "Tier 2"
        ],
        [160000, 70000],
        default=0
    )

    degree_bonus = np.select(
        [
            education_level == "MTech",
            education_level == "MBA",
            education_level == "MCA"
        ],
        [80000, 60000, 40000],
        default=0
    )

    placement_bonus = np.where(
        placement_status == "Placed",
        50000,
        -180000
    )

    salary = (
        180000
        + cgpa * 85000
        + internship_months * 18000
        + projects * 30000
        + certifications * 12000
        + skills_score * 7000
        + communication_score * 3500
        + tier_bonus
        + degree_bonus
        + placement_bonus
        + np.random.normal(0, 90000, n)
    )

    salary = np.clip(salary, 120000, 1800000)
    salary = salary.round(-3).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "education_level": education_level,
        "cgpa": cgpa,
        "internship_months": internship_months,
        "projects": projects,
        "certifications": certifications,
        "skills_score": skills_score,
        "communication_score": communication_score,
        "college_tier": college_tier,
        "placement_status": placement_status,
        "salary": salary
    })


# ============================================================
# 3. LOAD DATASET
# ============================================================

# Always initialize df before using it. If the CSV is missing, empty,
# unreadable, or does not contain the required salary target, use demo data.
REQUIRED_COLUMNS = {
    "age", "gender", "education_level", "cgpa", "internship_months",
    "projects", "certifications", "skills_score", "communication_score",
    "college_tier", "placement_status", "salary"
}

df = None
if os.path.exists(DATA_FILE):
    try:
        loaded_df = pd.read_csv(DATA_FILE)
        loaded_df = loaded_df.loc[:, ~loaded_df.columns.astype(str).str.match(r"^Unnamed")]
        missing_columns = REQUIRED_COLUMNS.difference(loaded_df.columns)
        if loaded_df.empty or missing_columns:
            print("CSV is empty or its columns do not match this project.")
            if missing_columns:
                print("Missing columns:", sorted(missing_columns))
            print("Using generated demo data instead.")
        else:
            df = loaded_df
            print("Existing dataset loaded.")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as error:
        print(f"Could not read dataset ({error}). Using generated demo data.")
else:
    print("Dataset not found. Creating demo dataset...")

if df is None:
    df = create_demo_dataset()
    # Keep the original mismatched CSV intact; save generated data separately
    # if a file already exists, otherwise create the standard dataset file.
    output_dataset = DATA_FILE if not os.path.exists(DATA_FILE) else os.path.join(BASE_DIR, "student_salary_demo.csv")
    df.to_csv(output_dataset, index=False)
    print(f"Demo dataset created: {output_dataset}")

# Ensure salary is numeric and remove rows where target salary is unavailable.
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
df = df.dropna(subset=["salary"]).copy()
if df.empty:
    raise ValueError("The dataset has no valid salary values. Check student_salary.csv.")


# ============================================================
# 4. BASIC DATA INFORMATION
# ============================================================

print("\n====================================")
print("DATASET INFORMATION")
print("====================================")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n====================================")
print("EXPLORATORY DATA ANALYSIS")
print("====================================")

plt.figure(figsize=(8, 5))
sns.histplot(df["salary"], kde=True, color="steelblue")
plt.title("Student Salary Distribution")
plt.xlabel("Annual Salary")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_FOLDER}/salary_distribution.png",
    dpi=150
)
plt.show()
plt.close()

numeric_columns = df.select_dtypes(
    include=np.number
).columns

plt.figure(figsize=(10, 7))
sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_FOLDER}/correlation_heatmap.png",
    dpi=150
)
plt.show()
plt.close()


# ============================================================
# 6. CREATE SALARY CLASSIFICATION
# ============================================================

print("\n====================================")
print("CREATING SALARY CATEGORIES")
print("====================================")

q1 = df["salary"].quantile(1 / 3)
q2 = df["salary"].quantile(2 / 3)


def salary_category(salary):
    if salary <= q1:
        return "Low"
    elif salary <= q2:
        return "Medium"
    return "High"


df["salary_category"] = df["salary"].apply(salary_category)

print(f"Low Salary      : <= ₹{q1:,.0f}")
print(f"Medium Salary   : ₹{q1:,.0f} - ₹{q2:,.0f}")
print(f"High Salary     : > ₹{q2:,.0f}")

plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="salary_category",
    order=["Low", "Medium", "High"],
    color="steelblue"
)
plt.title("Salary Category Distribution")
plt.xlabel("Salary Category")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_FOLDER}/salary_categories.png",
    dpi=150
)
plt.show()
plt.close()


# ============================================================
# 7. FEATURES AND TARGETS
# ============================================================

X = df.drop(columns=["salary", "salary_category"])

y_regression = df["salary"]
y_classification = df["salary_category"]


# ============================================================
# 8. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 9. PREPROCESSING
# ============================================================

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Support both newer and older scikit-learn versions.
try:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", one_hot_encoder)
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 10. TRAIN TEST SPLIT
# ============================================================

(
    X_train,
    X_test,
    y_reg_train,
    y_reg_test,
    y_clf_train,
    y_clf_test
) = train_test_split(
    X,
    y_regression,
    y_classification,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_classification
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 11. REGRESSION MODELS
# ============================================================

print("\n====================================")
print("REGRESSION")
print("====================================")

regression_models = {
    "Linear Regression": LinearRegression(),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Gradient Boosting Regressor":
        GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
            random_state=RANDOM_STATE
        )
}

regression_results = []
regression_pipelines = {}

for name, model in regression_models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_reg_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_reg_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_reg_test, predictions)
    )

    r2 = r2_score(y_reg_test, predictions)

    regression_results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    regression_pipelines[name] = pipeline

regression_results_df = pd.DataFrame(
    regression_results
).sort_values("R2", ascending=False)

print("\nRegression Results:")
print(regression_results_df.to_string(index=False))

regression_results_df.to_csv(
    f"{OUTPUT_FOLDER}/regression_results.csv",
    index=False
)


# ============================================================
# 12. BEST REGRESSION MODEL
# ============================================================

best_regression_name = regression_results_df.iloc[0]["Model"]

best_regression_model = regression_pipelines[
    best_regression_name
]

print("\nBEST REGRESSION MODEL:")
print(best_regression_name)

best_regression_prediction = (
    best_regression_model.predict(X_test)
)

plt.figure(figsize=(7, 6))

plt.scatter(
    y_reg_test,
    best_regression_prediction,
    alpha=0.65
)

minimum = min(
    y_reg_test.min(),
    best_regression_prediction.min()
)

maximum = max(
    y_reg_test.max(),
    best_regression_prediction.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_FOLDER}/actual_vs_predicted.png",
    dpi=150
)
plt.show()
plt.close()


# ============================================================
# 13. CLASSIFICATION MODELS
# ============================================================

print("\n====================================")
print("CLASSIFICATION")
print("====================================")

classification_models = {
    "Logistic Regression":
        LogisticRegression(max_iter=3000),

    "Random Forest Classifier":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Gradient Boosting Classifier":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
}

classification_results = []
classification_pipelines = {}

for name, model in classification_models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_clf_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_clf_test,
        predictions
    )

    precision = precision_score(
        y_clf_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_clf_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_clf_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    classification_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    classification_pipelines[name] = pipeline

classification_results_df = pd.DataFrame(
    classification_results
).sort_values("F1", ascending=False)

print("\nClassification Results:")
print(classification_results_df.to_string(index=False))

classification_results_df.to_csv(
    f"{OUTPUT_FOLDER}/classification_results.csv",
    index=False
)


# ============================================================
# 14. BEST CLASSIFICATION MODEL
# ============================================================

best_classification_name = (
    classification_results_df.iloc[0]["Model"]
)

best_classification_model = (
    classification_pipelines[best_classification_name]
)

print("\nBEST CLASSIFICATION MODEL:")
print(best_classification_name)

classification_prediction = (
    best_classification_model.predict(X_test)
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

class_labels = ["Low", "Medium", "High"]

cm = confusion_matrix(
    y_clf_test,
    classification_prediction,
    labels=class_labels
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Salary Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_FOLDER}/confusion_matrix.png",
    dpi=150
)
plt.show()
plt.close()

print("\nClassification Report:")

print(
    classification_report(
        y_clf_test,
        classification_prediction,
        labels=class_labels,
        zero_division=0
    )
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

print("\n====================================")
print("FEATURE IMPORTANCE")
print("====================================")

best_model = best_regression_model.named_steps["model"]

preprocessor_fitted = (
    best_regression_model.named_steps["preprocessor"]
)

feature_names = (
    preprocessor_fitted.get_feature_names_out()
)

if hasattr(best_model, "feature_importances_"):

    importance = best_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nTop factors affecting salary:")

    print(
        feature_importance.head(12).to_string(
            index=False
        )
    )

    feature_importance.to_csv(
        f"{OUTPUT_FOLDER}/feature_importance.csv",
        index=False
    )

    top_features = feature_importance.head(12)

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        color="steelblue"
    )

    plt.title("Top Factors Affecting Student Salary")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_FOLDER}/feature_importance.png",
        dpi=150
    )
    plt.show()
    plt.close()


# ============================================================
# 17. NEW STUDENT PREDICTION
# ============================================================

print("\n====================================")
print("NEW STUDENT PREDICTION")
print("====================================")

new_student = pd.DataFrame({
    "age": [22],
    "gender": ["Male"],
    "education_level": ["BTech"],
    "cgpa": [8.2],
    "internship_months": [6],
    "projects": [4],
    "certifications": [3],
    "skills_score": [82],
    "communication_score": [78],
    "college_tier": ["Tier 2"],
    "placement_status": ["Placed"]
})

predicted_salary = best_regression_model.predict(
    new_student
)[0]

predicted_category = best_classification_model.predict(
    new_student
)[0]

print("\nStudent Details:")
print(new_student.to_string(index=False))

print(
    f"\nPredicted Annual Salary: ₹{predicted_salary:,.0f}"
)

print(
    f"Predicted Salary Category: {predicted_category}"
)


# ============================================================
# 18. FINAL PROJECT SUMMARY
# ============================================================

print("\n====================================")
print("PROJECT SUMMARY")
print("====================================")

best_regression_row = regression_results_df.iloc[0]
best_classification_row = classification_results_df.iloc[0]

print(
    "\nBest Regression Model:",
    best_regression_row["Model"]
)

print(
    "R² Score:",
    round(best_regression_row["R2"], 4)
)

print(
    "MAE:",
    round(best_regression_row["MAE"], 2)
)

print(
    "RMSE:",
    round(best_regression_row["RMSE"], 2)
)

print(
    "\nBest Classification Model:",
    best_classification_row["Model"]
)

print(
    "Accuracy:",
    round(
        best_classification_row["Accuracy"] * 100,
        2
    ),
    "%"
)

print(
    "Precision:",
    round(
        best_classification_row["Precision"] * 100,
        2
    ),
    "%"
)

print(
    "Recall:",
    round(
        best_classification_row["Recall"] * 100,
        2
    ),
    "%"
)

print(
    "F1 Score:",
    round(
        best_classification_row["F1"] * 100,
        2
    ),
    "%"
)

print("\n====================================")
print("OUTPUT FILES")
print("====================================")

print(
    f"Graphs and result files are saved in: "
    f"{OUTPUT_FOLDER}/"
)

print("\nProject completed successfully! ✅")
