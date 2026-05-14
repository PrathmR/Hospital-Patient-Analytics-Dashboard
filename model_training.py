import pandas as pd
import numpy as np
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def generate_data(num_records=500):
    np.random.seed(42)
    
    genders = ['Male', 'Female']
    admission_types = ['Emergency', 'Elective', 'Urgent']
    severities = ['Minor', 'Moderate', 'Extreme']
    diagnosis_codes = ['Cardiology', 'Neurology', 'Oncology', 'Orthopedics', 'General']
    insurance_types = ['Medicare', 'Medicaid', 'Private', 'Self-Pay']
    departments = ['ICU', 'General Ward', 'Surgery', 'ER']
    
    data = {
        'PatientID': [f'P{i:04d}' for i in range(1, num_records + 1)],
        'Age': np.random.randint(18, 90, num_records),
        'Gender': np.random.choice(genders, num_records),
        'AdmissionType': np.random.choice(admission_types, num_records),
        'SeverityOfIllness': np.random.choice(severities, num_records),
        'NumLabProcedures': np.random.randint(1, 100, num_records),
        'DiagnosisCode': np.random.choice(diagnosis_codes, num_records),
        'InsuranceType': np.random.choice(insurance_types, num_records),
        'Department': np.random.choice(departments, num_records),
        'LOS': np.random.randint(1, 30, num_records),
    }
    
    df = pd.DataFrame(data)
    
    # Generate readmission risk based on some logic
    # Higher age, extreme severity, more lab procedures -> higher risk
    risk_score = (
        (df['Age'] / 90) * 0.3 +
        (df['SeverityOfIllness'] == 'Extreme').astype(int) * 0.4 +
        (df['NumLabProcedures'] / 100) * 0.2 +
        np.random.rand(num_records) * 0.2
    )
    
    df['Readmitted'] = (risk_score > 0.5).astype(int)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/patients_data.csv', index=False)
    print("Dataset generated and saved to 'data/patients_data.csv'")
    return df

def train_model(df):
    # Features for prediction tool
    # Target: Readmitted
    features = ['Age', 'Gender', 'AdmissionType', 'SeverityOfIllness', 'NumLabProcedures', 'DiagnosisCode', 'InsuranceType', 'Department']
    X = df[features]
    y = df['Readmitted']
    
    # Preprocessing
    numeric_features = ['Age', 'NumLabProcedures']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['Gender', 'AdmissionType', 'SeverityOfIllness', 'DiagnosisCode', 'InsuranceType', 'Department']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create Pipeline with RandomForest
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with test accuracy: {accuracy:.4f}")
    
    # Save model
    joblib.dump(model, 'model.pkl')
    print("Model saved to 'model.pkl'")

if __name__ == "__main__":
    df = generate_data(500)
    train_model(df)
