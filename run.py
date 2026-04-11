from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
import os

# ----------------------------
# SYMPTOM NORMALIZATION MAP
# ----------------------------
SYMPTOM_MAP = {
    "head pain": "headache",
    "stomach ache": "stomach pain",
    "tiredness": "fatigue",
    "weakness": "fatigue",
    "cold": "runny nose",
    "body pain": "body ache",
}

def normalize_symptom(symptom):
    symptom = symptom.lower().strip().replace("_", " ")
    return SYMPTOM_MAP.get(symptom, symptom)

# ----------------------------
# APP INIT
# ----------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = "your_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# LOAD MODEL + DATA
# ----------------------------
model = joblib.load(os.path.join(BASE_DIR, "app/ml_models/model1.pkl"))
symptom_list = joblib.load(os.path.join(BASE_DIR, "app/ml_models/symptoms.pkl"))

# Severity
severity_df = pd.read_csv(os.path.join(BASE_DIR, "app/data/Symptom-severity.csv"))
severity_df["Symptom"] = severity_df["Symptom"].str.lower().str.strip().str.replace("_", " ")
severity_dict = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# Description
desc_df = pd.read_csv(os.path.join(BASE_DIR, "app/data/symptom_Description.csv"))
desc_df["Disease"] = desc_df["Disease"].str.lower().str.strip()
desc_dict = dict(zip(desc_df["Disease"], desc_df["Description"]))

# Precautions
prec_df = pd.read_csv(os.path.join(BASE_DIR, "app/data/symptom_precaution.csv"))
prec_df["Disease"] = prec_df["Disease"].str.lower().str.strip()

prec_dict = {}
for _, row in prec_df.iterrows():
    precautions = [
        str(row[col]).strip()
        for col in prec_df.columns if "Precaution" in col and pd.notna(row[col])
    ]
    prec_dict[row["Disease"]] = precautions

# ----------------------------
# RULE ENGINE
# ----------------------------
def rule_based_checks(symptoms):
    symptoms = [s.lower() for s in symptoms]

    if "chest pain" in symptoms and "breathlessness" in symptoms:
        return "⚠️ Possible heart-related emergency. Seek immediate medical help."

    if "high fever" in symptoms and "vomiting" in symptoms:
        return "⚠️ Possible severe infection. Consult a doctor."

    if "unconsciousness" in symptoms:
        return "🚨 Emergency condition. Immediate attention required."

    return None

# ----------------------------
# VECTOR CREATION
# ----------------------------
def create_weighted_vector(input_symptoms):
    vector = [0] * len(symptom_list)

    for symptom in input_symptoms:
        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            weight = severity_dict.get(symptom, 1)
            vector[idx] = weight

    return vector

# ----------------------------
# EXPLANATION ENGINE
# ----------------------------
def generate_explanation(input_symptoms):
    explanation = []

    for symptom in input_symptoms:
        weight = severity_dict.get(symptom, 1)

        if weight >= 5:
            impact = "strong"
        elif weight >= 3:
            impact = "moderate"
        else:
            impact = "weak"

        explanation.append({
            "symptom": symptom,
            "impact": impact
        })

    return explanation

# ----------------------------
# TOP PREDICTIONS
# ----------------------------
def get_top_predictions(vector, input_symptoms, top_n=3):
    input_df = pd.DataFrame([vector], columns=symptom_list)
    probs = model.predict_proba(input_df)[0]
    top_indices = probs.argsort()[-top_n:][::-1]

    results = []

    for idx in top_indices:
        disease = model.classes_[idx]

        # ----------------------------
        # COVERAGE (REAL)
        # ----------------------------
        matched = sum([
            1 for s in input_symptoms if s in symptom_list
        ])
        

        coverage_score = matched / max(len(input_symptoms), 1)

        # ----------------------------
        # HYBRID SCORE (CLAMPED)
        # ----------------------------
        hybrid_score = (0.7 * probs[idx]) + (0.3 * coverage_score)
        hybrid_score = min(1.0, hybrid_score)

        # ----------------------------
        # BUILD RESULT
        # ----------------------------
        results.append({
            "disease": disease,
            "confidence": round(hybrid_score, 3),
            "raw_confidence": round(probs[idx], 3),
            "coverage": round(coverage_score, 2),
            "description": desc_dict.get(disease, "No description available"),
            "precautions": prec_dict.get(disease, ["No precautions available"]),
            "explanation": [
                {
                    "symptom": s,
                    "impact": (
                        "strong" if severity_dict.get(s, 1) >= 5
                        else "moderate" if severity_dict.get(s, 1) >= 3
                        else "weak"
                    )
                }
                for s in input_symptoms
            ]
        })

    # sort by confidence
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return results

# ----------------------------
# MAIN SYSTEM
# ----------------------------
def predict_system(input_symptoms):

    # normalize everything FIRST
    input_symptoms = [normalize_symptom(s) for s in input_symptoms]

    # rule check
    rule = rule_based_checks(input_symptoms)
    if rule:
        return {
            "type": "alert",
            "message": rule
        }

    vector = create_weighted_vector(input_symptoms)
    top_preds = get_top_predictions(vector, input_symptoms)

    if not top_preds:
        return {
            "type": "uncertain",
            "message": "No prediction could be made."
        }

    if top_preds[0]["confidence"] < 0.3:
        return {
            "type": "prediction",
            "results": top_preds,
            "note": "Low confidence prediction. Add more symptoms."
        }

    return {
        "type": "prediction",
        "results": top_preds
    }

# ----------------------------
# ROUTES (UNCHANGED)
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signin.html")
def signin():
    return render_template("signin.html")

@app.route("/signup.html")
def signup():
    return render_template("signup.html")

@app.route("/reminder.html")
def reminder():
    return render_template("reminder.html")

@app.route("/analysis.html", methods=["GET", "POST"])
def analysis():
    result = None
    selected_symptoms = []

    if request.method == "POST":
        raw = request.form.get("symptoms")

        if not raw:
            return render_template(
                "analysis.html",
                result={
                    "type": "uncertain",
                    "message": "No symptoms provided"
                },
                selected_symptoms=[]
            )

        selected_symptoms = [s.strip() for s in raw.split(",") if s.strip()]

        # 🔥 USE YOUR REAL SYSTEM
        result = predict_system(selected_symptoms)

    return render_template(
        "analysis.html",
        result=result,
        selected_symptoms=selected_symptoms
    )

# ----------------------------
# AUTOSUGGEST API
# ----------------------------
@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    query = request.args.get("q", "").lower()

    suggestions = [
        s for s in symptom_list
        if query in s
    ][:10]

    return jsonify(suggestions)

# ----------------------------
# AUTH ROUTES
# ----------------------------
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    if email == "test@test.com" and password == "123":
        session["user"] = email
        return redirect(url_for("analysis"))
    else:
        return "Invalid credentials", 401

@app.route("/register", methods=["POST"])
def register():
    return redirect(url_for("signin"))

# ----------------------------
# API ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", [])

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        result = predict_system(symptoms)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)