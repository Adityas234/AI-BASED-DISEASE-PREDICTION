from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = "your_secret_key"   # required for session handling

# ----------------------------
# Load ML model & vectorizer
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "app", "ml_models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "app", "ml_models", "vectorizer.pkl")
csv_path = os.path.join(BASE_DIR, "app", "data", "imdb_train_with_minor_disease.csv")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
disease_data = pd.read_csv(csv_path)

# ----------------------------
# Routes for frontend pages
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
    if request.method == "POST":
        symptoms = request.form.get("symptoms")
        if symptoms:
            X = vectorizer.transform([symptoms])
            predicted_disease = model.predict(X)[0]

            # Lookup in dataset
            info = disease_data[disease_data["disease"] == predicted_disease].to_dict(orient="records")
            if info:
                info = info[0]
                result = {
                    "major": predicted_disease,
                    "minor": info.get("minor_disease", "N/A"),
                    "precautions": info.get("precautions", "N/A"),
                    "medicines": info.get("medicine", "N/A"),
                }
            else:
                result = {
                    "major": predicted_disease,
                    "minor": "Not available",
                    "precautions": "Not available",
                    "medicines": "Not available",
                }

    return render_template("analysis.html", result=result)


# ----------------------------
# Auth routes (dummy for now)
# ----------------------------
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")
    # TODO: validate from DB
    if email == "test@test.com" and password == "123":
        session["user"] = email
        return redirect(url_for("analysis"))
    else:
        return "Invalid credentials", 401

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name") 
    email = request.form.get("email")
    password = request.form.get("password")
    # TODO: save to DB
    return redirect(url_for("signin"))


# ----------------------------
# Prediction API (analysis.html → model → analysis.html)
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        symptoms = request.form.get("symptoms")
        print("Received symptoms:", symptoms)

        if not symptoms or symptoms.strip() == "":
            return render_template("analysis.html", result=None)

        # 1️⃣ Vectorize input
        X = vectorizer.transform([symptoms])

        # 2️⃣ Predict disease
        predicted_disease = model.predict(X)[0]
        print("Predicted disease:", predicted_disease)

        # 3️⃣ Lookup details from dataset
        row = disease_data[
            disease_data["disease"].str.lower()
            == str(predicted_disease).lower()
        ]

        if row.empty:
            result = {
                "major": predicted_disease,
                "minor": "Not found in dataset",
                "precautions": "Consult a doctor",
                "medicines": "Consult a doctor"
            }
        else:
            result = {
                "major": predicted_disease,
                "minor": row.iloc[0]["minor_disease"],
                "precautions": row.iloc[0]["precautions"],
                "medicines": row.iloc[0]["medicines"]
            }

        return render_template("analysis.html", result=result)

    except Exception as e:
        print("❌ Prediction error:", e)
        return render_template("analysis.html", result=None)



@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        symptoms = request.form.get("symptoms")

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Vectorize
        X = vectorizer.transform([symptoms])

        # Predict disease
        predicted_disease = model.predict(X)[0]

        # Lookup in dataset
        row = disease_data[
            disease_data["disease"].str.lower()
            == str(predicted_disease).lower()
        ]

        if row.empty:
            return jsonify({
                "predicted_disease": predicted_disease,
                "minor_disease": None,
                "precautions": None,
                "medicines": None
            })

        return jsonify({
            "predicted_disease": predicted_disease,
            "minor_disease": row.iloc[0]["minor_disease"],
            "precautions": row.iloc[0]["precautions"],
            "medicines": row.iloc[0]["medicines"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
