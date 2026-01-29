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
    # Get symptoms text from frontend form
    symptoms = request.form.get("symptoms")

    if not symptoms:
        return render_template("analysis.html", prediction="Please enter symptoms")

    # Convert text using vectorizer
    X = vectorizer.transform([symptoms])

    # Predict using model
    prediction = model.predict(X)[0]

    # Render same page with result
    return render_template("analysis.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
