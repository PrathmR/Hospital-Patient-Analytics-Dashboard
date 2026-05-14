from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)

# Load Model
MODEL_PATH = 'model.pkl'
DATA_PATH = 'data/patients_data.csv'

# Helper to load data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html', active_page='dashboard')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', active_page='analytics')

@app.route('/predict')
def predict():
    return render_template('predict.html', active_page='predict')

@app.route('/api/kpis')
def api_kpis():
    df = load_data()
    if df.empty:
        return jsonify({})
    
    total_patients = len(df)
    avg_los = round(df['LOS'].mean(), 1)
    high_risk_readmission = int(df['Readmitted'].sum())
    
    # Department occupancy
    dept_counts = df['Department'].value_counts().to_dict()
    
    return jsonify({
        'total_patients': total_patients,
        'avg_los': avg_los,
        'high_risk_readmission': high_risk_readmission,
        'dept_occupancy': dept_counts
    })

@app.route('/api/patients')
def api_patients():
    df = load_data()
    if df.empty:
        return jsonify([])
    # Convert to list of dicts
    records = df.to_dict('records')
    return jsonify(records)

@app.route('/api/analytics')
def api_analytics():
    df = load_data()
    if df.empty:
        return jsonify({})
    
    # Generate some mock admission trend data based on the dataset size
    dates = [(datetime.now() - timedelta(days=30*i)).strftime('%Y-%b') for i in range(5, -1, -1)]
    admissions_trend = np.random.randint(50, 100, size=6).tolist()
    
    age_groups = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=['<30', '30-50', '50-70', '>70']).value_counts().to_dict()
    gender_dist = df['Gender'].value_counts().to_dict()
    insurance_dist = df['InsuranceType'].value_counts().to_dict()
    
    return jsonify({
        'trend': {'labels': dates, 'data': admissions_trend},
        'age': age_groups,
        'gender': gender_dist,
        'insurance': insurance_dist
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        age = float(data.get('age'))
        gender = data.get('gender')
        admission_type = data.get('admission_type')
        severity = data.get('severity')
        lab_procedures = float(data.get('lab_procedures'))
        diagnosis = data.get('diagnosis')
        insurance = data.get('insurance')
        department = data.get('department')
        
        # Load model
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Model not found. Please train the model first.'}), 500
            
        model = joblib.load(MODEL_PATH)
        
        # Create dataframe for prediction
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'AdmissionType': admission_type,
            'SeverityOfIllness': severity,
            'NumLabProcedures': lab_procedures,
            'DiagnosisCode': diagnosis,
            'InsuranceType': insurance,
            'Department': department
        }])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return jsonify({
            'risk_level': 'High' if prediction == 1 else 'Low',
            'probability': round(probability * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
