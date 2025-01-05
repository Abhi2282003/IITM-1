import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the datasets
credit_data = pd.read_excel('CreditWorthiness.xlsx', sheet_name='Data')
alt_data = pd.read_csv('alternative_dataset.csv')

# Streamlit UI
st.title("Loan Approval Prediction System")
st.write("## Problem Statement")
st.write("Real-Time Credit Risk Assessment Using Alternative Data")

st.write("## Objective")
st.write("Develop a system that assesses credit risk in real time by analyzing traditional financial data and alternative data sources like social media, utility payments, and spending habits.")

st.write("## Approach")
st.write("1. Aggregate and process alternative data sources alongside traditional credit information.")
st.write("2. Use machine learning models to predict creditworthiness.")
st.write("3. Provide lenders with transparent, explainable credit risk scores.")

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluation metrics in Streamlit
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
st.write("### Classification Report")
st.text(report)

# Visualization
st.write("### Visualizations")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart for loan decision distribution
loan_counts = combined_data['Loan_Decision'].value_counts()
ax[0].pie(loan_counts, labels=label_encoders['Loan_Decision'].classes_, autopct='%1.1f%%', startangle=140)
ax[0].set_title("Loan Decision Distribution")

# Feature importance bar plot
importances = model.feature_importances_
ax[1].barh(X.columns, importances)
ax[1].set_xlabel('Importance')
ax[1].set_ylabel('Features')
ax[1].set_title('Feature Importance for Loan Prediction')

st.pyplot(fig)

# Save model and encoders
joblib.dump(model, 'loan_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

uploaded_excel = st.file_uploader("Upload Credit Worthiness Data (Excel)", type=["xlsx"])
uploaded_csv = st.file_uploader("Upload Alternative Data (CSV)", type=["csv"])

if uploaded_excel is not None and uploaded_csv is not None:
    credit_data = pd.read_excel(uploaded_excel, sheet_name='Data')
    alt_data = pd.read_csv(uploaded_csv)
    combined_data = pd.concat([credit_data, alt_data], axis=1)

    for col in categorical_columns:
        if col in combined_data.columns:
            combined_data[col] = label_encoders[col].transform(combined_data[col].astype(str))
    
    predictions = model.predict(combined_data.drop('Loan_Decision', axis=1))
    combined_data['Loan_Prediction'] = label_encoders['Loan_Decision'].inverse_transform(predictions)

    st.write("### Predictions")
    st.dataframe(combined_data)

    st.write("### Visualization")
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(label_encoders['Loan_Decision'].classes_))
    plt.xticks(tick_marks, label_encoders['Loan_Decision'].classes_)
    plt.yticks(tick_marks, label_encoders['Loan_Decision'].classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)
