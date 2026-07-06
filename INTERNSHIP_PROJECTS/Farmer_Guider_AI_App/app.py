from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Crop Recommendation Model
crop_model = pickle.load(open('model/crop_recom_model.pkl', 'rb'))
crop_scaler = pickle.load(open('model/crop_recom_scaler.pkl', 'rb'))
crop_encoder = pickle.load(open('model/Crop_recom_encoder.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/crop', methods=['GET', 'POST'])
def crop():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = crop_scaler.transform(features)

        prediction = crop_model.predict(features_scaled)
        crop_name = crop_encoder.inverse_transform(prediction)[0]

        return render_template('result.html', result=f"Recommended Crop: {crop_name}")

    return render_template('crop.html')


if __name__ == '__main__':
    app.run(debug=True)