import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Load the datasets
credit_data = pd.read_excel('CreditWorthiness.xlsx', sheet_name='Data')
alt_data = pd.read_csv('alternative_dataset.csv')

# Streamlit UI
st.title("Loan Approval Prediction System")
st.write("## Real-Time Credit Risk Assessment Using Alternative Data")

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

# User Loan Enquiry Section
st.write("## Loan Enquiry Form")

with st.form(key='loan_form'):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    income = st.number_input("Monthly Income", min_value=0)
    employment_status = st.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Unemployed'])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    loan_amount = st.number_input("Loan Amount", min_value=1000)
    loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Education', 'Business', 'Medical'])
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        # Randomly select alternative data
        random_alt_data = alt_data.sample(n=1).reset_index(drop=True)

        # Combine user input with random alternative data
        user_data = pd.DataFrame(random_alt_data)
        user_data['Age'] = age
        user_data['Income'] = income
        user_data['Employment_Status'] = employment_status
        user_data['Credit_Score'] = credit_score
        user_data['Loan_Amount'] = loan_amount
        user_data['Loan_Purpose'] = loan_purpose
        
        # Display user submitted data with random alternative data
        st.write("### Combined User Data and Alternative Data")
        st.dataframe(user_data)
        
        # Encode user data
        for col in user_data.columns:
            if col in label_encoders:
                user_data[col] = label_encoders[col].transform(user_data[col].astype(str))
        
        # Align user data with training features
        user_data = user_data.reindex(columns=feature_names, fill_value=0)
        
        # Predict loan eligibility and probabilities
        prediction = model.predict(user_data)
        prediction_proba = model.predict_proba(user_data)
        loan_result = label_encoders['Loan_Decision'].inverse_transform(prediction)
        
        # Extract feature importances and provide reason for denial
        if loan_result[0] == 'Denied':
            importances = model.feature_importances_
            most_important_features = np.argsort(importances)[-3:][::-1]
            reasons = feature_names[most_important_features]
            st.write(f"### Loan Prediction for {name}: Denied")
            st.write("#### Top Reasons for Denial:")
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write(f"### Loan Prediction for {name}: Approved")
