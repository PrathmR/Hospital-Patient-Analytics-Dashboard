# Hospital Patient Analytics Dashboard

A production-ready Flask application that provides actionable insights into patient data using Machine Learning. The dashboard helps healthcare professionals monitor KPIs, analyze admission trends, and predict patient readmission risk.

## 🚀 Features

- **KPI Dashboard**: High-level overview of Total Patients, Average Length of Stay (LOS), High-Risk Readmission Count, and Department-wise occupancy.
- **Advanced Analytics**: Interactive charts showing admission trends and demographic breakdowns (Age, Gender, Insurance Type).
- **ML Risk Prediction**: A tool that uses a Random Forest model to predict the 30-day readmission risk based on 8 patient metrics.
- **Patient Roster**: A searchable, color-coded list of all patients with their associated risk levels.
- **Responsive Design**: Built with Bootstrap 5 and a "Healthcare Blue" palette, optimized for both desktop and mobile.

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualizations**: Chart.js

## 📁 Project Structure

```text
├── app.py              # Flask application routes and logic
├── model_training.py   # Script to generate synthetic data and train the ML model
├── model.pkl           # Trained Random Forest model (generated after training)
├── data/
│   └── patients_data.csv # Synthetic patient dataset
├── static/
│   ├── css/
│   │   └── style.css   # Custom styling and theme
│   └── js/
│       └── main.js    # Frontend interactivity
├── templates/
│   ├── base.html       # Base layout with sidebar
│   ├── index.html      # Dashboard home
│   ├── analytics.html  # Analytics and charts page
│   └── predict.html    # ML Prediction tool
└── requirements.txt    # Python dependencies
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Final_Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the ML Model**:
   This will generate the synthetic dataset and the trained model file.
   ```bash
   python model_training.py
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Dashboard**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

## 🧠 ML Model Details

The model is a **Random Forest Classifier** trained on synthetic data. It takes the following features as input:
- Age
- Gender
- Admission Type
- Severity of Illness
- Number of Lab Procedures
- Diagnosis Category
- Insurance Type
- Department

The model provides a "High" or "Low" readmission risk along with a confidence probability.
