from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model/model2.1.pkl', 'rb') as f:
    model2 = pickle.load(f)

with open('model/scaler (1).pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    person_age = float(request.form['person_age'])
    person_income = float(request.form['person_income'])
    person_emp_length = float(request.form['person_emp_length'])
    loan_amnt = float(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    loan_percent_income = float(request.form['loan_percent_income'])
    cb_person_cred_hist_length = float(request.form['cb_person_cred_hist_length'])
    
    # Placeholder values for now — we'll fix these once encoding is sorted
    person_home_ownership = 0
    loan_intent = 0
    loan_grade = 0
    cb_person_default_on_file = 0

    features = np.array([[person_age, person_income, person_home_ownership,
                           person_emp_length, loan_intent, loan_grade,
                           loan_amnt, loan_int_rate, loan_percent_income,
                           cb_person_default_on_file, cb_person_cred_hist_length]])
    
    features_scaled = scaler.transform(features)
    prediction = model2.predict(features_scaled)
    
    return render_template("result.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)