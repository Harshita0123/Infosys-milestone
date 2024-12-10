import streamlit as st
import numpy as np
import joblib  # Import joblib to load the pre-trained model
import pandas as pd  # Import pandas to work with DataFrame

class LogisticRegression:
    # Initialize learning rate (lr) and number of iterations (n_iters)
    def _init_(self, lr=0.001, n_iters=1000):
        self.lr = lr   # Learning rate (α)
        self.n_iters = n_iters  # Number of iterations for gradient descent
        self.weights = None  # Initialize weights (slopes)
        self.bias = None  # Initialize bias (intercept)

    # Sigmoid function to map predictions to probabilities between 0 and 1
    def sigmoid(self, z):
        # Applying the sigmoid function: y_pred = 1 / (1 + exp(-z))
        return 1 / (1 + np.exp(-z))

    # Compute the cost function (Log Loss or Binary Cross-Entropy)
    def compute_cost(self, X, y):
        m = len(y)  # Number of training samples
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)  # Predicted probabilities
        # Log loss cost function
        cost = - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost

    # Function to train the logistic regression model using X_train and y_train
    def fit(self, X, y): 
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero

        # Perform gradient descent to optimize weights and bias
        for i in range(self.n_iters):
            
            # Calculate the linear combination of inputs and weights: z = X * weights + bias
            model = np.dot(X, self.weights) + self.bias
            
            # Transforming the linear output into probabilities using the sigmoid function
            y_pred = self.sigmoid(model)

            # Gradient calculation for the weights (dw): 
            # dw = (1 / num_samples) * X.T * (y_pred - y)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y)) 
            
            # Gradient calculation for the bias (db):
            # db = (1 / num_samples) * sum(y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y) 

            # Update the weights and bias using the gradients and learning rate:
            # weights := weights - learning_rate * dw, bias := bias - learning_rate * db
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # print the cost every 100 iterations
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {i}, Cost: {cost}")
        
        # Final cost after training
        print(f"Final Cost after {self.n_iters} iterations: {cost}")

    # Function to predict labels based on input test data (X_test)
    def predict(self, X):
        # Calculate the linear output from the input data and learned weights: z = X * weights + bias
        linear_model = np.dot(X, self.weights) + self.bias 
        
        # Transform the linear output into predicted probabilities using the sigmoid function
        y_pred = self.sigmoid(linear_model)  # Applying sigmoid to get probabilities: y_pred = 1 / (1 + exp(-z))

        # Classify based on threshold 0.5: if probability >= 0.5, classify as 1, else 0
        predictions = []
        for p in y_pred:
            if p >= 0.5:
                predictions.append(1)  # Classify as 1 if probability is >= 0.5
            else:
                predictions.append(0)  # Classify as 0 if probability is < 0.5

        return np.array(predictions)  # Return the final binary predictions as an array
    
    # Function to calculate accuracy as the proportion of correct predictions using X_test and y_test
    def score(self, X, y):
        # Predict labels for the input data
        y_pred = self.predict(X) 
        
        # Calculate accuracy as the percentage of correct predictions
        accuracy = np.mean(y_pred == y)  # Accuracy = (correct predictions) / (total predictions)
        
        return accuracy 

    # Function to return predicted probabilities for the positive class (class 1) using X_test
    def predict_proba(self, X):
        # Compute the linear combination of inputs and weights
        linear_model = np.dot(X, self.weights) + self.bias 

        # Get probability of positive class (class 1)
        probabilities = self.sigmoid(linear_model) 

        # Return both probabilities (negative class, positive class) 
        return [(1 - p, p) for p in probabilities]

log_reg_model = joblib.load('log_reg_model.pkl')

# Streamlit app
st.title('Stroke Prediction')

st.write('---')

# Inputs for the stroke prediction model
age = st.slider('Age of the individual', 0, 120, 50)
hypertension = st.radio('Hypertension (0 = No, 1 = Yes)', (0, 1))
heart_disease = st.radio('Heart Disease (0 = No, 1 = Yes)', (0, 1))
ever_married = st.radio('Ever Married (0 = No, 1 = Yes)', (0, 1))
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=0.0, step=0.1)
bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, step=0.1)
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

# Additional Inputs for gender (as per dataset)
gender = st.radio('Gender (0 = Female, 1 = Male)', (0, 1))

# Create a DataFrame to store the input features (with 17 features but no 'id')
input_data = pd.DataFrame({
    'gender': [gender],  # Gender (if part of the model)
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'residence_type': [1 if residence_type == 'Urban' else 0],  # 1 for Urban, 0 for Rural
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'work_type_Govt_job': [1 if work_type == 'Govt_job' else 0],
    'work_type_Never_worked': [1 if work_type == 'Never_worked' else 0],
    'work_type_Private': [1 if work_type == 'Private' else 0],
    'work_type_Self-employed': [1 if work_type == 'Self-employed' else 0],
    'work_type_children': [1 if work_type == 'Children' else 0],
    'smoking_status_Unknown': [1 if smoking_status == 'Unknown' else 0],
    'smoking_status_formerly smoked': [1 if smoking_status == 'formerly smoked' else 0],
    'smoking_status_never smoked': [1 if smoking_status == 'never smoked' else 0],
    'smoking_status_smokes': [1 if smoking_status == 'smokes' else 0]
})

# Add a button for prediction
if st.button('Predict Stroke Risk'):
    # Pass the DataFrame to the pre-trained model's predict_proba method
    risk_probability = log_reg_model.predict_proba(input_data)

    # Get the stroke risk percentage (second column is the probability of stroke)
    risk_percentage = risk_probability[0][1] * 100

    # Display the result
    st.markdown(f'<p style="font-size:20px;">Predicted Stroke Risk: {risk_percentage:.2f}%</p>', unsafe_allow_html=True)