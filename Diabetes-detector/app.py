# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Injecting custom CSS to style the app
st.markdown("""
    <style>
        .main {
            background-color: #f1f2f6;
        }

        .css-18e3th9 {
            text-align: center;
            color: #2d3436;
            font-size: 35px;
            font-family: 'Arial', sans-serif;
        }

        .css-1d391kg {
            background-color: #ffffff;
        }

        .css-1emrehy {
            background-color: #0984e3;
            color: white;
            font-weight: bold;
        }

        .css-1cpxqw2 {
            border: 2px solid #0984e3;
            border-radius: 5px;
            padding: 8px;
        }

        .stAlert {
            background-color: #ffcccc;
            color: #b33939;
            font-weight: bold;
        }

        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the web app
st.title("Diabetes Detection using Support Vector Machine (SVM)")

# Description and instructions
st.markdown("""
This web app uses a Support Vector Machine (SVM) model to predict whether an individual has diabetes or not based on their medical data.
Fill in the form with your data, and the model will give you a prediction.

**Features to be entered**:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
""")

# File upload option
uploaded_file = st.file_uploader("/content/diabetes.csv", type=["csv"])

# Global scaler definition (outside any function to access in both training and prediction)
scaler = StandardScaler()

# Initialize the model variable
model = None

# Handling file upload
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.write(data.head())

    if 'Outcome' in data.columns:
        X = data.drop(columns=['Outcome'])
        y = data['Outcome']
    else:
        st.error("CSV file doesn't contain a valid 'Outcome' column.")
        st.stop()

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
    X_test_scaled = scaler.transform(X_test)  # Transform test data

    # Train the SVM model
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Display classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Visualizing Confusion Matrix
    cm = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(cm.figure)

# Sidebar for user input
st.sidebar.header("Enter your medical data:")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 50, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 30, 150, 80)
skin_thickness = st.sidebar.slider("Skin Thickness", 10, 70, 20)
insulin = st.sidebar.slider("Insulin", 10, 300, 50)
bmi = st.sidebar.slider("BMI", 10, 50, 25)
diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
age = st.sidebar.slider("Age", 18, 100, 30)

# Create a dataframe for user input
user_input = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]], 
                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Ensure that the model is trained before making predictions
if model is not None:
    # Scale the user input using the scaler fitted on the training data
    user_input_scaled = scaler.transform(user_input)  # Use the fitted scaler to transform user input

    # Make a prediction based on user input
    user_prediction = model.predict(user_input_scaled)

    # Display the prediction
    if user_prediction[0] == 0:
        st.success("You are **NOT** likely to have diabetes.")
    else:
        st.warning("You are **LIKELY** to have diabetes.")

    # Visualize user input data
    st.subheader("Your Data:")
    st.write(user_input)

else:
    st.error("Model is not trained yet. Please upload a dataset and train the model first.")

# Adding a footer
st.markdown("---")
st.markdown("Built with ❤️ by [Chahat,Harsh,Sanjeevani,Trapti].")
