from flask import Flask, render_template, request
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# ---------------- LOAD MODELS ----------------
lr = pickle.load(open("linear_model.pkl", "rb"))
log_reg = pickle.load(open("logistic_model.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))

# ---------------- LOAD SCALER & COLUMNS ----------------
scaler = pickle.load(open("scaler.pkl", "rb"))
columns_lr  = pickle.load(open("columns_lr.pkl", "rb"))
columns_log = pickle.load(open("columns_log.pkl", "rb"))
columns_rf  = pickle.load(open("columns_rf.pkl", "rb"))

# ---------------- DROPDOWN AUTO-GENERATION ----------------
def extract_categories(columns, prefix):
    return set(col.replace(prefix, "") for col in columns if col.startswith(prefix))

# combine all columns safely
all_columns = set(columns_lr) | set(columns_log) | set(columns_rf)

dropdowns = {
    "gender": ["Male", "Female"],

    "academic": ["High School" , "Undergraduate" , "Graduate"],
    "country": sorted(extract_categories(all_columns, "Country_")),
    "platform": sorted(extract_categories(all_columns, "Most_Used_Platform_")),
    "relationship": ["Single" , "In Relationship" , "Complicated"]
}

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- COMMON INPUT PROCESSOR ----------------
def process_input(form, columns):
    df = pd.DataFrame(0, columns=columns, index=[0])

    # Numerical
    df["Age"] = float(form["age"])
    df["Avg_Daily_Usage_Hours"] = float(form["usage"])
    df["Mental_Health_Score"] = float(form["mental"])

    df[["Age", "Avg_Daily_Usage_Hours", "Mental_Health_Score"]] = scaler.transform(
        df[["Age", "Avg_Daily_Usage_Hours", "Mental_Health_Score"]]
    )

    # One-hot categorical
    def set_col(col):
        if col in df.columns:
            df[col] = 1

    set_col(f"Gender_{form['gender']}")
    set_col(f"Academic_Level_{form['academic']}")
    set_col(f"Country_{form['country']}")
    set_col(f"Most_Used_Platform_{form['platform']}")
    set_col(f"Relationship_Status_{form['relationship']}")

    return df[columns]

# ---------------- LINEAR REGRESSION ----------------
@app.route("/addiction-score", methods=["GET", "POST"])
def addiction_score():
    prediction = None
    if request.method == "POST":
        X = process_input(request.form, columns_lr)
        prediction = round(lr.predict(X)[0], 2)

    return render_template(
        "addiction_score.html",
        prediction=prediction,
        dropdowns=dropdowns
    )

# ---------------- LOGISTIC REGRESSION ----------------
@app.route("/academic-impact", methods=["GET", "POST"])
def academic_impact():
    prediction = None
    if request.method == "POST":
        X = process_input(request.form, columns_log)
        result = log_reg.predict(X)[0]
        prediction = "Yes" if result == 1 else "No"

    return render_template(
        "academic_impact.html",
        prediction=prediction,
        dropdowns=dropdowns
    )

# ---------------- RANDOM FOREST ----------------
@app.route("/high-addiction", methods=["GET", "POST"])
def high_addiction():
    prediction = None
    if request.method == "POST":
        X = process_input(request.form, columns_rf)
        result = rf.predict(X)[0]
        prediction = "High Addiction" if result == 1 else "Low Addiction"

    return render_template(
        "high_addiction.html",
        prediction=prediction,
        dropdowns=dropdowns
    )

if __name__ == "__main__":
    app.run(debug=True)
