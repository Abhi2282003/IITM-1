import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import time

# Load the datasets
credit_data = pd.read_excel('CreditWorthiness.xlsx', sheet_name='Data')
alt_data = pd.read_csv('alternative_dataset.csv')

# Streamlit UI
st.set_page_config(page_title="Loan Approval System", page_icon="💼", layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.write("### Automated Loan Application Assessment")

# Merge datasets (assuming equal lengths and synthetic alignment)
combined_data = pd.concat([credit_data, alt_data], axis=1)

# Encode categorical variables
label_encoders = {}
categorical_columns = combined_data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = combined_data.drop('Loan_Decision', axis=1)
y = combined_data['Loan_Decision']

# Save feature names for validation
feature_names = X.columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'loan_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

# Custom CSS for professional UI
st.markdown("""
    <style>
    .reportview-container {
        background: url("https://source.unsplash.com/featured/?finance,business") no-repeat center fixed;
        background-size: cover;
    }
    .css-18e3th9 {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# User Loan Enquiry Section
st.write("## 📋 Loan Application Form")

with st.form(key='loan_form'):
    name = st.text_input("Name")
    age = st.slider("Age", min_value=18, max_value=100, step=1)
    income = st.number_input("Monthly Income", min_value=0)
    employment_status = st.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Unemployed'])
    loan_amount = st.number_input("Loan Amount", min_value=1000)
    loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Education', 'Business', 'Medical'])
    submit_button = st.form_submit_button("Submit Application")

    if submit_button:
        # Simulate spinner effect with animation
        with st.spinner('⏳ Processing your application...'):
            time.sleep(3)
        
        # Randomly select alternative data
        random_alt_data = alt_data.sample(n=1).reset_index(drop=True)

        # Combine user input with random alternative data
        user_data = pd.DataFrame(random_alt_data)
        user_data['Age'] = age
        user_data['Income'] = income
        user_data['Employment_Status'] = employment_status
        user_data['Loan_Amount'] = loan_amount
        user_data['Loan_Purpose'] = loan_purpose
        
        # Display user submitted data with random alternative data
        st.write("### 🧾 Application Summary")
        st.dataframe(user_data)
        
        # Encode user data
        for col in user_data.columns:
            if col in label_encoders:
                user_data[col] = label_encoders[col].transform(user_data[col].astype(str))
        
        # Align user data with training features
        user_data = user_data.reindex(columns=feature_names, fill_value=0)
        
        # Predict loan eligibility
        prediction = model.predict(user_data)
        loan_result = label_encoders['Loan_Decision'].inverse_transform(prediction)
        
        # Display prediction result
        st.write("### Loan Prediction Result")
        if loan_result[0] == 'Denied':
            st.error(f"❌ Loan Prediction for {name}: Denied")
        else:
            st.success(f"✅ Loan Prediction for {name}: Approved")
