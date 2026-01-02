from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/predict')
def predict_form():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def predict():
    age = float(request.form['age'])
    gender = int(request.form['gender'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    ap_hi = float(request.form['ap_hi'])
    ap_lo = float(request.form['ap_lo'])
    chol = int(request.form['chol'])
    gluc = int(request.form['gluc'])
    smoke = int(request.form['smoke'])
    alco = int(request.form['alco'])
    active = int(request.form['active'])
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo

    features_to_scale = np.array([[height, weight, ap_hi, ap_lo, age, pulse_pressure, bmi]])

    scaled_features = scaler.transform(features_to_scale)

    s_height = scaled_features[0][0]
    s_weight = scaled_features[0][1]
    s_ap_hi = scaled_features[0][2]
    s_ap_lo = scaled_features[0][3]
    s_age = scaled_features[0][4]
    s_pp = scaled_features[0][5]
    s_bmi = scaled_features[0][6]

    final_input = np.array([[gender, s_height, s_weight, s_ap_hi, s_ap_lo, chol, gluc, smoke, alco, active, s_age, s_pp, s_bmi]])

    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0][1]

    if prediction[0] == 1:
        result_text = 'High Risk of Cardiovascular Disease'
        color = 'danger'  # Bootstrap red
    else:
        result_text = 'Low Risk (Healthy)'
        color = 'success'  # Bootstrap green

    return render_template('result.html', prediction_text=result_text, prob=round(probability * 100, 2), color=color)

if __name__ == '__main__':
    app.run(debug=True)