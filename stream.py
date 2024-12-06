import streamlit as st
import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None  
        self.bias = None      

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            error = y_predicted - y
            dw = (1 / num_samples) * np.dot(X.T, error)
            db = (1 / num_samples) * np.sum(error)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

# Load data and train the model (this would typically be done once)
np.random.seed(0)
X = np.random.rand(100, 9)  # 100 samples, 9 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # A simple pattern for demonstration

log_reg_model = LogisticRegressionFromScratch(learning_rate=0.01, iterations=1000)
log_reg_model.fit(X, y)

# Streamlit app
st.title('Stroke Prediction')

st.write('---')

# Inputs for the stroke prediction model
age = st.slider('Age of the individual', 0, 120, 50)
hypertension = st.radio('Hypertension (0 = No, 1 = Yes)', (0, 1))
heart_disease = st.radio('Heart Disease (0 = No, 1 = Yes)', (0, 1))
ever_married = st.radio('Ever Married (0 = No, 1 = Yes)', (0, 1))
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=0.0, step=0.1)
bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, step=0.1)
smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

# Map categorical inputs to numerical values
work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Children': 3, 'Never_worked': 4}
residence_type_mapping = {'Urban': 0, 'Rural': 1}
smoking_status_mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}

def predict_proba(features):
    return log_reg_model.predict_proba(features)

if st.button('Predict Stroke Risk'):
    features = np.array([[age, hypertension, heart_disease, ever_married, 
                          work_type_mapping[work_type], residence_type_mapping[residence_type], 
                          avg_glucose_level, bmi, smoking_status_mapping[smoking_status]]])
    risk_probability = predict_proba(features)
    risk_percentage = risk_probability[0] * 100
    st.markdown(f'<p style="font-size:20px;">Predicted Stroke Risk: {risk_percentage:.2f}%</p>', unsafe_allow_html=True)
