import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI-Powered Loan Approval",
    page_icon="💰",
    layout="wide",
)

# -------------------- UTILITY FUNCTIONS --------------------
def load_datasets():
    """
    Load two datasets and combine them by stacking rows (axis=0).
    You must have 'CreditWorthiness.xlsx' and 'alternative_dataset.csv'
    in the same folder as this app, or change paths accordingly.
    """
    df_excel = pd.read_excel("CreditWorthiness.xlsx", sheet_name="Data")
    df_alt = pd.read_csv("alternative_dataset.csv")

    # Combine them row-wise
    combined_df = pd.concat([df_excel, df_alt], axis=0, ignore_index=True)
    return combined_df


def encode_categorical(df):
    """
    Label-encode object-type columns (including Loan_Decision).
    """
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def train_and_save_model():
    """Train a RandomForest model on 'Loan_Decision' and save artifacts."""
    df = load_datasets()

    if "Loan_Decision" not in df.columns:
        st.warning("'Loan_Decision' column not found. Cannot train model.")
        return None, None, None

    # Drop rows where Loan_Decision is missing
    df.dropna(subset=["Loan_Decision"], inplace=True)

    # Optional: Print distribution (for debugging)
    st.write("Distribution of Loan_Decision BEFORE encoding:")
    st.write(df["Loan_Decision"].value_counts(dropna=False))

    # Encode categorical fields (including target)
    df, le_dict = encode_categorical(df)

    # Check distribution after encoding
    st.write("Distribution of Loan_Decision AFTER encoding:")
    st.write(df["Loan_Decision"].value_counts(dropna=False))

    # Split into features and target
    X = df.drop("Loan_Decision", axis=1)
    y = df["Loan_Decision"]
    feature_names = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, 
        random_state=42, 
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write("Model Training Completed.")
    st.write(f"Accuracy: {acc:.2f}")
    st.write("Classification Report:\n", report)
    st.write("Confusion Matrix:\n", cm)

    # Save artifacts
    joblib.dump(model, "loan_prediction_model.pkl")
    joblib.dump(le_dict, "label_encoders.pkl")
    joblib.dump(feature_names, "feature_names.pkl")

    return model, le_dict, feature_names


def load_model_artifacts():
    """
    Load trained model, label encoders, and feature names from disk.
    If they don't exist, returns (None, None, None).
    """
    try:
        model = joblib.load("loan_prediction_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        features = joblib.load("feature_names.pkl")
        return model, encoders, features
    except:
        return None, None, None


def init_applications():
    """
    Load or initialize the applications.csv file.
    This file tracks user-submitted loan applications.
    """
    cols = [
        "Application_ID", "Username", "Name", 
        "Age", "Income", "Employment_Status", 
        "Loan_Amount", "Loan_Purpose", "Status", 
        "Submission_Time"
    ]
    try:
        df = pd.read_csv("applications.csv", dtype={"Application_ID": str})
        # Ensure required columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]
    except FileNotFoundError:
        return pd.DataFrame(columns=cols)


def save_applications(df):
    """Save the applications DataFrame to CSV."""
    df.to_csv("applications.csv", index=False)


def generate_application_id():
    """Generate a unique ID with a timestamp."""
    return datetime.now().strftime("APP-%Y%m%d%H%M%S%f")


# -------------- PDF GENERATION USING REPORTLAB (for Streamlit Cloud) --------------
def generate_pdf_report(application_data):
    """
    Generate a simple PDF loan report using ReportLab.
    Returns PDF as bytes (in-memory).
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Some basic styling
    text_x = 1 * inch
    y = 10 * inch
    line_height = 0.3 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(text_x, y, "Loan Application Report")
    y -= line_height * 2

    c.setFont("Helvetica", 12)

    # Helper to print lines
    def print_line(label, value):
        nonlocal y
        c.drawString(text_x, y, f"{label}: {value}")
        y -= line_height

    # Print fields
    print_line("Application ID", application_data.get("Application_ID", "N/A"))
    print_line("Name", application_data.get("Name", "N/A"))
    print_line("Age", application_data.get("Age", "N/A"))
    print_line("Income", application_data.get("Income", "N/A"))
    print_line("Employment Status", application_data.get("Employment_Status", "N/A"))
    print_line("Loan Amount", application_data.get("Loan_Amount", "N/A"))
    print_line("Loan Purpose", application_data.get("Loan_Purpose", "N/A"))
    print_line("Current Status", application_data.get("Status", "N/A"))
    print_line("Submission Time", application_data.get("Submission_Time", "N/A"))

    c.showPage()
    c.save()
    buffer.seek(0)

    return buffer.getvalue()


# -------------------- LOAD/VERIFY MODEL --------------------
model, label_encoders, feature_names = load_model_artifacts()
if not model:
    st.info("No existing model found. Training a new one...")
    model, label_encoders, feature_names = train_and_save_model()

# -------------------- SESSION STATE for Applications --------------------
if "applications" not in st.session_state:
    st.session_state.applications = init_applications()

# -------------------- SIMPLE AUTH --------------------
st.sidebar.title("User Authentication")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

# Track login state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_type = None
    st.session_state.username = None

if login_button:
    # Simple check for demonstration
    if username == "user" and password == "123":
        st.session_state.authenticated = True
        st.session_state.user_type = "customer"
        st.session_state.username = username
    elif username == "admin" and password == "123":
        st.session_state.authenticated = True
        st.session_state.user_type = "bank"
        st.session_state.username = username
    else:
        st.sidebar.error("Invalid username or password!")

# -------------------- MAIN APP --------------------
if st.session_state.authenticated:
    # Branding header
    st.markdown(
        """
        <div style="text-align:center; background-color:#F7F7F7; 
                    padding:20px; border-radius:10px; margin-bottom:20px;">
            <h1 style="color:#2E86C1; margin-bottom:0;">AI-Powered Loan Approval</h1>
            <p style="font-size:18px; margin:0; color:#000000;">
                Leverage machine learning for fast, reliable loan decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------- CUSTOMER PORTAL --------------------
    if st.session_state.user_type == "customer":
        st.subheader(f"Welcome, {st.session_state.username}!")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Submit a New Loan Application")
            with st.form("loan_form"):
                name = st.text_input("Full Name")
                age = st.slider("Age", 18, 100, 30)
                income = st.number_input("Monthly Income", min_value=0, value=5000)
                employment_status = st.selectbox(
                    "Employment Status", 
                    ["Employed", "Self-Employed", "Unemployed"]
                )
                loan_amount = st.number_input("Loan Amount", min_value=1000, value=10000)
                loan_purpose = st.selectbox(
                    "Loan Purpose", 
                    ["Home", "Education", "Business", "Medical"]
                )

                submit_button = st.form_submit_button("Submit Application")

            if submit_button:
                # Generate unique ID
                app_id = generate_application_id()
                submit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                new_row = pd.DataFrame({
                    "Application_ID": [app_id],
                    "Username": [st.session_state.username],
                    "Name": [name],
                    "Age": [age],
                    "Income": [income],
                    "Employment_Status": [employment_status],
                    "Loan_Amount": [loan_amount],
                    "Loan_Purpose": [loan_purpose],
                    "Status": ["Pending"],
                    "Submission_Time": [submit_time]
                })

                # Add to session state and save
                st.session_state.applications = pd.concat(
                    [st.session_state.applications, new_row],
                    ignore_index=True
                )
                save_applications(st.session_state.applications)
                st.success(f"Application submitted! Your ID: {app_id}")

        with col2:
            st.markdown("### Your Applications")
            # Filter only the logged-in user's apps
            user_apps = st.session_state.applications[
                st.session_state.applications["Username"] == st.session_state.username
            ].copy()

            # Sort by most recent
            user_apps["Submission_Time"] = pd.to_datetime(user_apps["Submission_Time"], errors="coerce")
            user_apps.sort_values("Submission_Time", ascending=False, inplace=True)

            # Show in a table
            st.dataframe(user_apps.drop(["Username"], axis=1), use_container_width=True)

    # -------------------- BANK PORTAL --------------------
    else:
        st.subheader("Bank Portal - Manage Applications")
        apps_df = st.session_state.applications.copy()
        apps_df.fillna("", inplace=True)

        if apps_df.empty:
            st.info("No applications found.")
        else:
            # Dropdown to pick an application
            selected_app_id = st.selectbox(
                "Select Application ID", 
                apps_df["Application_ID"].unique()
            )

            if selected_app_id:
                # Get the row for the selected application
                app_row = apps_df[apps_df["Application_ID"] == selected_app_id].iloc[0]

                st.write("#### Application Details")
                st.table(app_row)

                # Generate Prediction
                if model is not None and st.button("Generate Prediction"):
                    # Prepare user_data
                    drop_cols = [
                        "Name", "Status", "Username", 
                        "Application_ID", "Submission_Time"
                    ]
                    user_data = app_row.drop(drop_cols, errors="ignore").to_frame().T.copy()

                    # Apply label encoders
                    for col in user_data.columns:
                        if col in label_encoders:
                            user_data[col] = label_encoders[col].transform(
                                user_data[col].astype(str)
                            )

                    # Ensure columns match training features
                    user_data = user_data.reindex(columns=feature_names, fill_value=0)

                    prediction = model.predict(user_data)

                    # If the target was label-encoded, decode it
                    if "Loan_Decision" in label_encoders:
                        loan_result = label_encoders["Loan_Decision"].inverse_transform(prediction)
                    else:
                        loan_result = prediction

                    suggested_status = loan_result[0]
                    st.write(f"**Model Suggests:** {suggested_status}")

                # Generate & Download PDF Report
                if st.button("View / Generate Report"):
                    pdf_bytes = generate_pdf_report(app_row.to_dict())

                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"LoanReport_{selected_app_id}.pdf",
                        mime="application/pdf"
                    )

                # Update Final Status
                st.markdown("---")
                st.write("### Update Status")
                possible_statuses = ["Pending", "Approved", "Denied"]
                current_status = app_row["Status"] if app_row["Status"] in possible_statuses else "Pending"

                final_status = st.selectbox(
                    "Set Final Status", 
                    possible_statuses,
                    index=possible_statuses.index(current_status) 
                        if current_status in possible_statuses else 0
                )
                if st.button("Confirm Status"):
                    st.session_state.applications.loc[
                        st.session_state.applications["Application_ID"] == selected_app_id,
                        "Status"
                    ] = final_status
                    save_applications(st.session_state.applications)
                    st.success(f"Application {selected_app_id} updated to '{final_status}'")

else:
    st.warning("Please log in to access the system.")
