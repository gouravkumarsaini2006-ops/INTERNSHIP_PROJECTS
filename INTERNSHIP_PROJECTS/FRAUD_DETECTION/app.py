from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

print("Loading model files...")
try:
    with open('model/encoder.pkl(k-means)', 'rb') as f:
        encoders = pickle.load(f)
    with open('model/scaler.pkl(k-means)', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/kmeans (1).pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Catch Form Data
            transaction_amount = float(request.form['Transaction_Amount'])
            transaction_type = request.form['Transaction_Type']
            device_used = request.form['Device_Used']
            location = request.form['Location']
            payment_method = request.form['Payment_Method']
            
            # 2. DataFrame
            input_data = pd.DataFrame([[
                transaction_amount, transaction_type, device_used, location, payment_method
            ]], columns=['Transaction_Amount', 'Transaction_Type', 'Device_Used', 'Location', 'Payment_Method'])
            
            # 3. Encode
            for col in ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']:
                if input_data[col].iloc[0] in encoders[col].classes_:
                     input_data[col] = encoders[col].transform(input_data[col])
                else:
                     input_data[col] = encoders[col].transform(['Unknown'])

            # 4. Scale & Predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Terminal Logger
            print(f"\n--- TRANSACTION LOG ---")
            print(f"Amount: ${transaction_amount}")
            print(f"Result: {'🚨 FRAUD (1)' if prediction == 1 else '✅ SAFE (0)'}")
            print(f"-----------------------\n")
            
            return render_template('result.html', prediction=int(prediction))
            
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)