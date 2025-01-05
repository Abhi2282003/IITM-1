import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from weasyprint import HTML  # For converting HTML to PDF in memory
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
    Load and combine your two datasets by stacking rows 
    (axis=0) rather than side by side. 
    """
    df_excel = pd.read_excel("CreditWorthiness.xlsx", sheet_name="Data")
    df_alt = pd.read_csv("alternative_dataset.csv")

    # IMPORTANT: If both files have the same columns 
    # or very similar ones, we can concat row-wise.
    combined_df = pd.concat([df_excel, df_alt], axis=0, ignore_index=True)

    # If the two datasets truly have different sets of columns,
    # you may need a different merging strategy. This is the 
    # typical approach when building a bigger training dataset.

    return combined_df

def encode_categorical(df):
    """Label-encode object-type columns (excluding the target if desired)."""
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        # If you want to exclude Loan_Decision from being label-encoded,
        # remove the `if col != 'Loan_Decision':` comment below
        # if col != 'Loan_Decision':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def train_and_save_model():
    """Train a model on 'Loan_Decision' and save artifacts."""
    df = load_datasets()

    # Check that 'Loan_Decision' column exists and is not all NaN
    if "Loan_Decision" not in df.columns:
        st.warning("'Loan_Decision' column not found. Cannot train model.")
        return None, None, None

    # Drop rows where Loan_Decision is missing
    df.dropna(subset=["Loan_Decision"], inplace=True)

    # Print distribution of the target to confirm we have multiple classes
    print("Distribution of Loan_Decision BEFORE encoding:")
    print(df["Loan_Decision"].value_counts(dropna=False))

    # Encode categorical fields (including Loan_Decision)
    df, le_dict = encode_categorical(df)

    # Confirm distribution after encoding
    print("Distribution of Loan_Decision AFTER encoding:")
    print(df["Loan_Decision"].value_counts(dropna=False))

    # Now define X, y
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

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Training Completed.")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # Save artifacts
    joblib.dump(model, "loan_prediction_model.pkl")
    joblib.dump(le_dict, "label_encoders.pkl")
    joblib.dump(feature_names, "feature_names.pkl")

    return model, le_dict, feature_names

def load_model_artifacts():
    """Load trained model, label encoders, and feature names from disk."""
    try:
        model = joblib.load("loan_prediction_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        features = joblib.load("feature_names.pkl")
        return model, encoders, features
    except:
        return None, None, None

def init_applications():
    """Load or initialize the applications.csv."""
    cols = [
        "Application_ID", "Username", "Name", 
        "Age", "Income", "Employment_Status", 
        "Loan_Amount", "Loan_Purpose", "Status", 
        "Submission_Time"
    ]
    try:
        df = pd.read_csv("applications.csv", dtype={"Application_ID": str})
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

# =============== HTML REPORT GENERATION & PDF (WEASYPRINT) ===============
def generate_report_html(application_data):
    # Extract fields from application_data (a Series or dict)
    app_id = application_data.get("Application_ID", "N/A")
    name = application_data.get("Name", "N/A")
    age = application_data.get("Age", "N/A")
    income = application_data.get("Income", "N/A")
    emp_status = application_data.get("Employment_Status", "N/A")
    loan_amount = application_data.get("Loan_Amount", "N/A")
    loan_purpose = application_data.get("Loan_Purpose", "N/A")
    status = application_data.get("Status", "N/A")
    submission_time = application_data.get("Submission_Time", "N/A")

    # Example “progress” values to mimic bar chart visuals:
    doc_collection_percent = 70
    verification_percent = 50
    underwriting_percent = 30

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Loan Report - {app_id}</title>
      <style>
        body {{
          margin: 0; padding: 0;
          font-family: Arial, sans-serif;
          background-color: #0f3933;
        }}
        .report-container {{
          width: 800px;
          margin: 40px auto;
          background-color: #ffffff;
          padding: 30px 40px;
          border-radius: 8px;
        }}
        .header {{
          display: flex; justify-content: space-between; align-items: flex-start;
          margin-bottom: 30px;
        }}
        .header-left h1 {{
          font-size: 28px;
          color: #016a5b;
          margin: 0 0 10px 0;
        }}
        .header-left p {{
          margin: 0; color: #666; line-height: 1.5;
        }}
        .header-right h2 {{
          font-size: 16px;
          margin: 0; text-transform: uppercase; color: #016a5b;
        }}
        .header-right p {{
          margin: 5px 0 0 0; color: #999; font-size: 14px;
        }}
        h3.section-title {{
          font-size: 20px; color: #016a5b; margin-bottom: 10px;
        }}
        .progress-container {{
          margin: 20px 0;
        }}
        .progress-bar {{
          display: flex; align-items: center; margin-bottom: 10px;
        }}
        .progress-bar-label {{
          width: 150px; font-size: 14px; color: #333;
        }}
        .progress-bar-track {{
          flex: 1; background-color: #e5e5e5; height: 10px; margin: 0 10px;
          border-radius: 5px; position: relative;
        }}
        .progress-bar-fill {{
          background-color: #016a5b; height: 100%; border-radius: 5px;
          transition: width 0.3s ease-in-out;
        }}
        .progress-bar-value {{
          width: 40px; font-size: 14px; color: #333; text-align: right;
        }}
        .info-block {{
          margin: 20px 0;
        }}
        .info-block p {{
          margin: 6px 0; line-height: 1.5; color: #555;
        }}
        .footer {{
          margin-top: 30px; text-align: center; font-size: 12px; color: #666;
        }}
      </style>
    </head>
    <body>
      <div class="report-container">
        <!-- Header -->
        <div class="header">
          <div class="header-left">
            <h1>Loan Application Report</h1>
            <p>Application ID: {app_id}</p>
          </div>
          <div class="header-right">
            <h2>Your Bank Name</h2>
            <p>{datetime.now().strftime("%d/%m/%Y")}</p>
          </div>
        </div>

        <!-- Applicant Info -->
        <h3 class="section-title">Applicant Information</h3>
        <div class="info-block">
          <p><strong>Name:</strong> {name}</p>
          <p><strong>Age:</strong> {age}</p>
          <p><strong>Monthly Income:</strong> {income}</p>
          <p><strong>Employment Status:</strong> {emp_status}</p>
          <p><strong>Loan Amount Requested:</strong> {loan_amount}</p>
          <p><strong>Loan Purpose:</strong> {loan_purpose}</p>
          <p><strong>Current Status:</strong> {status}</p>
          <p><strong>Submission Time:</strong> {submission_time}</p>
        </div>

        <!-- Progress Section -->
        <h3 class="section-title">Progress</h3>
        <p>This section shows hypothetical progress for certain key steps 
           (Document Collection, Verification, Underwriting) as an example.</p>
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-bar-label">Document Collection</div>
            <div class="progress-bar-track">
              <div class="progress-bar-fill" style="width: {doc_collection_percent}%;"></div>
            </div>
            <div class="progress-bar-value">{doc_collection_percent}%</div>
          </div>

          <div class="progress-bar">
            <div class="progress-bar-label">Verification</div>
            <div class="progress-bar-track">
              <div class="progress-bar-fill" style="width: {verification_percent}%;"></div>
            </div>
            <div class="progress-bar-value">{verification_percent}%</div>
          </div>

          <div class="progress-bar">
            <div class="progress-bar-label">Underwriting</div>
            <div class="progress-bar-track">
              <div class="progress-bar-fill" style="width: {underwriting_percent}%;"></div>
            </div>
            <div class="progress-bar-value">{underwriting_percent}%</div>
          </div>
        </div>

        <!-- Next Steps -->
        <h3 class="section-title">Next Steps</h3>
        <div class="info-block">
          <p>1. Complete pending verifications (income, identity).</p>
          <p>2. Finalize underwriting decision.</p>
          <p>3. Notify applicant of final approval/denial.</p>
        </div>

        <!-- Footer -->
        <div class="footer">
          &copy; {datetime.now().year} Your Bank Name | Confidential
        </div>
      </div>
    </body>
    </html>
    """
    return html_content

def html_to_pdf(html_string):
    """Convert the given HTML string to PDF bytes using WeasyPrint."""
    pdf_bytes = HTML(string=html_string).write_pdf()
    return pdf_bytes

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

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_type = None
    st.session_state.username = None

if login_button:
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
    # Header / Branding
    st.markdown(
        """
        <div style="text-align:center; background-color:#F7F7F7; padding:20px; border-radius:10px; margin-bottom:20px;">
            <h1 style="color:#2E86C1; margin-bottom:0;">AI-Powered Loan Approval</h1>
            <p style="font-size:18px; margin:0; color:#000000;">Leverage machine learning for fast, reliable loan decisions.</p>
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
                employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
                loan_amount = st.number_input("Loan Amount", min_value=1000, value=10000)
                loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Business", "Medical"])

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

                st.session_state.applications = pd.concat(
                    [st.session_state.applications, new_row],
                    ignore_index=True
                )
                save_applications(st.session_state.applications)
                st.success(f"Application submitted! Your ID: {app_id}")

        with col2:
            st.markdown("### Your Applications")
            user_apps = st.session_state.applications[
                st.session_state.applications["Username"] == st.session_state.username
            ].copy()

            # Sort by submission time descending
            user_apps["Submission_Time"] = pd.to_datetime(user_apps["Submission_Time"], errors="coerce")
            user_apps.sort_values("Submission_Time", ascending=False, inplace=True)

            st.dataframe(user_apps.drop(["Username"], axis=1), use_container_width=True)

    # -------------------- BANK PORTAL --------------------
    else:
        st.subheader("Bank Portal - Manage Applications")
        apps_df = st.session_state.applications.copy()
        apps_df.fillna("", inplace=True)

        if apps_df.empty:
            st.info("No applications found.")
        else:
            # Display a dropdown of all application IDs
            selected_app_id = st.selectbox("Select Application ID", apps_df["Application_ID"].unique())

            if selected_app_id:
                app_row = apps_df[apps_df["Application_ID"] == selected_app_id].iloc[0]

                st.write("#### Application Details")
                st.table(app_row)

                # 1. Generate Prediction Button
                if model is not None and st.button("Generate Prediction"):
                    # Prepare user_data for prediction
                    drop_cols = ["Name", "Status", "Username", "Application_ID", "Submission_Time"]
                    user_data = app_row.drop(drop_cols, errors="ignore").to_frame().T.copy()

                    # Apply label encoders (for columns that the model expects)
                    for col in user_data.columns:
                        if col in label_encoders:
                            user_data[col] = label_encoders[col].transform(user_data[col].astype(str))

                    # Reindex to match training features exactly
                    user_data = user_data.reindex(columns=feature_names, fill_value=0)

                    prediction = model.predict(user_data)

                    # If the target was label-encoded, decode it
                    if "Loan_Decision" in label_encoders:
                        loan_result = label_encoders["Loan_Decision"].inverse_transform(prediction)
                    else:
                        loan_result = prediction

                    suggested_status = loan_result[0]
                    st.write(f"**Model Suggests:** {suggested_status}")

                # 2. View/Generate PDF Report
                if st.button("View / Generate Report"):
                    # Generate the HTML
                    html_report = generate_report_html(app_row.to_dict())

                    # Show the HTML inline
                    with st.expander("View Loan Report (HTML Preview)", expanded=True):
                        st.components.v1.html(html_report, height=700, scrolling=True)

                    # Convert HTML to PDF with WeasyPrint
                    pdf_bytes = html_to_pdf(html_report)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"LoanReport_{selected_app_id}.pdf",
                        mime="application/pdf"
                    )

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
