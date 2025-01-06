import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from datetime import datetime
import os

from weasyprint import HTML  # For converting HTML to PDF in memory
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="AI-Powered Loan Approval",
    page_icon="💰",
    layout="wide",
)

# -------------------- FEATURE DEFINITIONS --------------------
# Define all categorical features used in both alternative and creditworthiness data
CATEGORICAL_FEATURES = [
    # Alternative categorical features
    "Social_Media_Activity",
    "Utility_Payment_Timeliness",
    "Geolocation_Stability",
    
    # Creditworthiness categorical features
    "Chist",
    "Cpur",
    "MSG",
    "Prop",
    "inPlans",
    "Htype",
    "JobType",
    "telephone",
    "foreign"
]

# Define all numerical features with their respective ranges
NUMERICAL_FEATURES = {
    # Alternative numerical features
    "Online_Shopping_Frequency": [0, 10],
    "Mobile_Usage_Hours": [0, 12],
    "App_Subscriptions": [0, 20],
    "Streaming_Subscriptions": [0, 5],
    "Ecommerce_Transactions": [0, 15],
    "Ride_Sharing_Usage": [0, 30],
    "Smart_Device_Usage_Hours": [0, 24],
    "Digital_Payment_Transactions": [0, 50],
    
    # Creditworthiness numerical features
    "Cbal": [100, 5000],
    "Cdur": [1, 60],
    "Camt": [1000, 50000],
    "Sbal": [0, 5000],
    "Edur": [0, 12],
    "InRate": [0, 30],
    "Oparties": [0, 5],
    "Rdur": [0, 30],
    "age": [18, 90],
    "NumCred": [0, 10],
    "Ndepend": [0, 5],
    "creditScore": [300, 850]
}

# -------------------- UTILITY FUNCTIONS --------------------
def standardize_categorical(df):
    """
    Convert all categorical features to lowercase and strip whitespace to ensure consistency.
    """
    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns:
            df[feature] = df[feature].astype(str).str.lower().str.strip()
    return df

def ensure_all_categories(df, feature, categories):
    """
    Ensure that all specified categories are present in the DataFrame for a given feature.
    If any categories are missing, add synthetic rows to include them.
    """
    existing_categories = df[feature].dropna().unique().tolist()
    missing_categories = [cat for cat in categories if cat not in existing_categories]
    if missing_categories:
        # Create synthetic rows for missing categories
        synthetic_rows = pd.DataFrame({feature: missing_categories})
        # For other features, assign default or NaN values
        for col in df.columns:
            if col != feature:
                if df[col].dtype == 'object':
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    synthetic_rows[col] = mode_value
                else:
                    synthetic_rows[col] = 0
        df = pd.concat([df, synthetic_rows], ignore_index=True)
    return df

def encode_categorical(df):
    """
    Label-encode all categorical (object-type) columns and return the encoders.
    """
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def load_datasets():
    """
    Load and combine the two datasets: 'CreditWorthiness.xlsx' and 'alternative_dataset.csv'.
    Ensure that all categorical data is standardized.
    """
    try:
        df_excel = pd.read_excel("CreditWorthiness.xlsx", sheet_name="Data")
    except FileNotFoundError:
        st.error("Error: 'CreditWorthiness.xlsx' not found in the application directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading 'CreditWorthiness.xlsx': {e}")
        st.stop()
        
    try:
        df_alt = pd.read_csv("alternative_dataset.csv")
    except FileNotFoundError:
        st.error("Error: 'alternative_dataset.csv' not found in the application directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading 'alternative_dataset.csv': {e}")
        st.stop()

    # Combine datasets
    combined_df = pd.concat([df_excel, df_alt], axis=0, ignore_index=True)
    combined_df = standardize_categorical(combined_df)  # Standardize categorical data
    return combined_df

def train_and_save_model():
    """
    Train the RandomForestClassifier model on the combined dataset and save the model,
    label encoders, and feature names for future use.
    """
    df = load_datasets()

    # Define all possible categories for each categorical feature
    categorical_features_dict = {
        "Social_Media_Activity": ["low", "medium", "high"],
        "Utility_Payment_Timeliness": ["on_time", "late", "irregular"],
        "Geolocation_Stability": ["stable", "somewhat_stable", "unstable"],
        "Chist": ["clean", "delinquent", "no_history"],
        "Cpur": ["car", "house", "personal", "business"],
        "MSG": ["single", "married", "divorced", "widowed"],
        "Prop": ["owner", "renter", "other"],
        "inPlans": ["yes", "no"],
        "Htype": ["urban", "suburban", "rural"],
        "JobType": ["employed", "self_employed", "unemployed"],
        "telephone": ["yes", "no"],
        "foreign": ["yes", "no"]
        # Add other categorical features and their possible categories here if needed
    }

    # Ensure all categories are present in the DataFrame
    for feature, categories in categorical_features_dict.items():
        if feature in df.columns:
            df = ensure_all_categories(df, feature, categories)
        else:
            st.error(f"Error: Feature '{feature}' not found in the dataset.")
            st.stop()

    # Check that 'Loan_Decision' column exists and is not all NaN
    if "Loan_Decision" not in df.columns:
        st.error("Error: 'Loan_Decision' column not found in the dataset.")
        st.stop()

    # Drop rows where 'Loan_Decision' is missing
    df.dropna(subset=["Loan_Decision"], inplace=True)

    st.write("### Distribution of Loan_Decision BEFORE Encoding")
    st.write(df["Loan_Decision"].value_counts(dropna=False))

    # Encode categorical features
    df, le_dict = encode_categorical(df)

    st.write("### Distribution of Loan_Decision AFTER Encoding")
    st.write(df["Loan_Decision"].value_counts(dropna=False))

    # Separate features and target
    X = df.drop("Loan_Decision", axis=1)
    y = df["Loan_Decision"]
    feature_names = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write("### Model Training Completed")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Classification Report:**")
    st.text(report)
    st.write("**Confusion Matrix:**")
    st.write(cm)

    # Save model and encoders
    try:
        joblib.dump(model, "loan_prediction_model.pkl")
        joblib.dump(le_dict, "label_encoders.pkl")
        joblib.dump(feature_names, "feature_names.pkl")
        st.success("Model and encoders have been successfully trained and saved.")
    except Exception as e:
        st.error(f"Error saving model artifacts: {e}")
        st.stop()

    return model, le_dict, feature_names

def load_model_artifacts():
    """
    Load the trained model, label encoders, and feature names from disk.
    If any artifacts are missing, return None for each.
    """
    try:
        model = joblib.load("loan_prediction_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        features = joblib.load("feature_names.pkl")
        return model, encoders, features
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

def init_applications():
    """
    Load existing loan applications from 'applications.csv' or initialize a new DataFrame.
    """
    cols = [
        "Application_ID", "Username", "Name", 
        "Age", "Income", "Employment_Status", 
        "Loan_Amount", "Loan_Purpose", "Status", 
        "Submission_Time"
    ]
    if os.path.exists("applications.csv"):
        try:
            df = pd.read_csv("applications.csv", dtype={"Application_ID": str})
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df[cols]
        except Exception as e:
            st.error(f"Error loading 'applications.csv': {e}")
            st.stop()
    else:
        return pd.DataFrame(columns=cols)

def save_applications(df):
    """
    Save the loan applications DataFrame to 'applications.csv'.
    """
    try:
        df.to_csv("applications.csv", index=False)
    except Exception as e:
        st.error(f"Error saving 'applications.csv': {e}")

def generate_application_id():
    """
    Generate a unique application ID based on the current timestamp.
    """
    return datetime.now().strftime("APP-%Y%m%d%H%M%S%f")

def generate_report_html(application_data, feature_data):
    """
    Generate an HTML report for the loan application using descriptive feature values.
    
    :param application_data: Dictionary containing application details.
    :param feature_data: Dictionary containing generated feature values.
    """
    app_id = application_data.get("Application_ID", "N/A")
    name = application_data.get("Name", "N/A")
    age = application_data.get("Age", "N/A")
    income = application_data.get("Income", "N/A")
    emp_status = application_data.get("Employment_Status", "N/A")
    loan_amount = application_data.get("Loan_Amount", "N/A")
    loan_purpose = application_data.get("Loan_Purpose", "N/A")
    status = application_data.get("Status", "N/A")
    submission_time = application_data.get("Submission_Time", "N/A")
    
    # Retrieve feature data with proper formatting
    def get_feature_display(feature_name):
        value = feature_data.get(feature_name, "N/A")
        if feature_name in CATEGORICAL_FEATURES:
            return value.replace("_", " ").capitalize()
        return value

    social_media_activity = get_feature_display("Social_Media_Activity")
    utility_payment_timeliness = get_feature_display("Utility_Payment_Timeliness")
    geolocation_stability = get_feature_display("Geolocation_Stability")
    chist = get_feature_display("Chist")
    cpur = get_feature_display("Cpur")
    msg = get_feature_display("MSG")
    prop = get_feature_display("Prop")
    inPlans = get_feature_display("inPlans")
    htype = get_feature_display("Htype")
    jobType = get_feature_display("JobType")
    telephone = get_feature_display("telephone")
    foreign = get_feature_display("foreign")
    
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
          background-color: #f2f2f2;
        }}
        .report-container {{
          width: 800px;
          margin: 40px auto;
          background-color: #ffffff;
          padding: 30px 40px;
          border-radius: 8px;
          box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
          display: flex; justify-content: space-between; align-items: flex-start;
          margin-bottom: 30px;
        }}
        .header-left h1 {{
          font-size: 28px;
          color: #2E86C1;
          margin: 0 0 10px 0;
        }}
        .header-left p {{
          margin: 0; color: #555; line-height: 1.5;
        }}
        .header-right h2 {{
          font-size: 16px;
          margin: 0; text-transform: uppercase; color: #2E86C1;
        }}
        .header-right p {{
          margin: 5px 0 0 0; color: #888; font-size: 14px;
        }}
        h3.section-title {{
          font-size: 20px; color: #2E86C1; margin-bottom: 10px;
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
          background-color: #2E86C1; height: 100%; border-radius: 5px;
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
          margin-top: 30px; text-align: center; font-size: 12px; color: #888;
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
          <p><strong>Monthly Income:</strong> ${income}</p>
          <p><strong>Employment Status:</strong> {emp_status}</p>
          <p><strong>Loan Amount Requested:</strong> ${loan_amount}</p>
          <p><strong>Loan Purpose:</strong> {loan_purpose}</p>
          <p><strong>Current Status:</strong> {status}</p>
          <p><strong>Submission Time:</strong> {submission_time}</p>
        </div>

        <!-- Generated Features -->
        <h3 class="section-title">Generated Features</h3>
        <div class="info-block">
          <p><strong>Social Media Activity:</strong> {social_media_activity}</p>
          <p><strong>Utility Payment Timeliness:</strong> {utility_payment_timeliness}</p>
          <p><strong>Geolocation Stability:</strong> {geolocation_stability}</p>
          <p><strong>Credit History:</strong> {chist}</p>
          <p><strong>Credit Purpose:</strong> {cpur}</p>
          <p><strong>Marital Status:</strong> {msg}</p>
          <p><strong>Property Ownership:</strong> {prop}</p>
          <p><strong>In Plans:</strong> {inPlans}</p>
          <p><strong>Housing Type:</strong> {htype}</p>
          <p><strong>Job Type:</strong> {jobType}</p>
          <p><strong>Telephone Ownership:</strong> {telephone}</p>
          <p><strong>Foreign:</strong> {foreign}</p>
        </div>

        <!-- Progress Section -->
        <h3 class="section-title">Progress</h3>
        <p>This section shows the hypothetical progress of your loan application.</p>
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
    """
    Convert the given HTML string to PDF bytes using WeasyPrint.
    """
    try:
        pdf_bytes = HTML(string=html_string).write_pdf()
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# -------------------- APPLICATION FEATURES HANDLING --------------------
def init_application_features():
    """
    Load existing application features from 'application_features.csv' or initialize a new DataFrame.
    """
    if os.path.exists("application_features.csv"):
        try:
            df = pd.read_csv("application_features.csv", dtype={"Application_ID": str})
            st.success("Loaded existing application features.")
            return df
        except Exception as e:
            st.error(f"Error loading 'application_features.csv': {e}")
            st.stop()
    else:
        # Initialize an empty DataFrame with Application_ID and all feature columns
        feature_columns = CATEGORICAL_FEATURES + list(NUMERICAL_FEATURES.keys())
        cols = ["Application_ID"] + feature_columns
        df = pd.DataFrame(columns=cols)
        st.info("'application_features.csv' not found. Initialized a new DataFrame for application features.")
        return df

def save_application_features(df):
    """
    Save the application features DataFrame to 'application_features.csv'.
    """
    try:
        df.to_csv("application_features.csv", index=False)
    except Exception as e:
        st.error(f"Error saving 'application_features.csv': {e}")

# -------------------- MODEL ARTIFACTS LOADING --------------------
model, label_encoders, feature_names = load_model_artifacts()
if not model:
    st.info("No existing model found. Training a new one...")
    model, label_encoders, feature_names = train_and_save_model()
    if not model:
        st.stop()

# -------------------- APPLICATIONS DATA INITIALIZATION --------------------
if "applications" not in st.session_state:
    st.session_state.applications = init_applications()

# -------------------- APPLICATION FEATURES INITIALIZATION --------------------
if "application_features" not in st.session_state:
    st.session_state.application_features = init_application_features()

# -------------------- SIMPLE AUTHENTICATION --------------------
st.sidebar.title("User Authentication")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_type = None
    st.session_state.username = None

if login_button:
    if username.lower() == "user" and password == "123":
        st.session_state.authenticated = True
        st.session_state.user_type = "customer"
        st.session_state.username = username
        st.sidebar.success("Logged in as Customer.")
    elif username.lower() == "admin" and password == "123":
        st.session_state.authenticated = True
        st.session_state.user_type = "bank"
        st.session_state.username = username
        st.sidebar.success("Logged in as Bank Admin.")
    else:
        st.sidebar.error("Invalid username or password!")

# -------------------- MAIN APPLICATION --------------------
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
                name = st.text_input("Full Name", max_chars=100)
                age = st.slider("Age", 18, 100, 30)
                income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=500)
                employment_status = st.selectbox(
                    "Employment Status", 
                    ["Employed", "Self-Employed", "Unemployed"]
                )
                loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=10000, step=1000)
                loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Business", "Medical"])
                
                # -------- TERMS AND CONDITIONS -------------
                st.markdown("""---""")
                st.markdown("#### Terms & Conditions")
                
                st.write("""
                **User Consent Request for Data Access**

                Dear Applicant,

                To provide you with enhanced services and a more personalized experience, we are requesting 
                your permission to access specific data through our secure APIs. The data will help us tailor our 
                offerings to meet your unique needs. Below is an outline of the data we wish to access and how we 
                ensure the highest level of security, compliance, and privacy:

                **Types of Data We Wish to Access:**
                - **Transaction Data:** Information related to your financial transactions.
                - **Social Media Activity Data:** Frequency of posts and interactions.
                - **Geolocation Data:** Location-based information to enhance relevance of services.
                - **App Subscription Data:** Data about your app subscriptions for tailored experiences.
                - **Online Shopping Data:** Information about your online shopping behavior for personalized 
                  product recommendations.
                - **Other Relevant Data:** Any other data necessary to improve our services.

                **How We Ensure Your Privacy and Security:**
                - **Authorization:** Explicit consent requested before accessing any data.
                - **Data Security:** All data is encrypted both in transit and at rest.
                - **Compliance:** Strict adherence to GDPR, CCPA, and other relevant laws.
                - **Access Limitations:** Only necessary data is accessed.
                - **Transparency:** You will be notified whenever we access your data, and you can review or 
                  delete your information at any time.

                By providing consent, you allow us to access and process the specified data for the purposes 
                mentioned above. We guarantee that your data will be handled securely, and your privacy is 
                our top priority.
                """)

                consent_given = st.checkbox(
                    "I agree to allow [Your Company Name] to access my data as outlined above."
                )

                submit_button = st.form_submit_button("Submit Application")

            if submit_button:
                if not consent_given:
                    st.warning("You must accept the Terms and Conditions to submit the loan application.")
                elif not name.strip():
                    st.warning("Please enter your full name.")
                else:
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
                    st.success(f"Application submitted successfully! Your Application ID: **{app_id}**")

        with col2:
            st.markdown("### Your Applications")
            user_apps = st.session_state.applications[
                st.session_state.applications["Username"] == st.session_state.username
            ].copy()

            if not user_apps.empty:
                # Convert 'Submission_Time' to datetime and sort
                user_apps["Submission_Time"] = pd.to_datetime(user_apps["Submission_Time"], errors="coerce")
                user_apps.sort_values("Submission_Time", ascending=False, inplace=True)

                # Display applications excluding 'Username'
                st.dataframe(user_apps.drop(["Username"], axis=1), use_container_width=True)
            else:
                st.info("You have no applications submitted yet.")

    # -------------------- BANK PORTAL --------------------
    else:
        st.subheader("Bank Portal - Manage Applications")
        apps_df = st.session_state.applications.copy()
        apps_df.fillna("", inplace=True)

        if apps_df.empty:
            st.info("No applications found.")
        else:
            selected_app_id = st.selectbox("Select Application ID", apps_df["Application_ID"].unique())

            if selected_app_id:
                app_row = apps_df[apps_df["Application_ID"] == selected_app_id].iloc[0]

                st.write("#### Application Details")
                st.table(app_row)

                # 1. Generate Prediction Button
                if model is not None:
                    # Use a unique key for the button to prevent multiple triggers
                    if st.button("Generate Prediction", key="predict_btn"):
                        # Check if features already exist for this application
                        features_df = st.session_state.application_features
                        if selected_app_id in features_df["Application_ID"].values:
                            # Retrieve existing features
                            app_features = features_df[features_df["Application_ID"] == selected_app_id].iloc[0].to_dict()
                            st.success("Using existing features for prediction.")
                        else:
                            # ---------------------- RANDOM DATA GENERATION ----------------------
                            random_data = {"Application_ID": selected_app_id}
                            
                            # Generate random data for categorical features using LabelEncoder's classes_
                            for feature in CATEGORICAL_FEATURES:
                                if feature in label_encoders:
                                    # Select a random category from known classes
                                    random_category = random.choice(label_encoders[feature].classes_)
                                    random_data[feature] = random_category
                                else:
                                    st.error(f"Error: No LabelEncoder found for categorical feature '{feature}'.")
                                    st.stop()
                            
                            # Generate random data for numerical features within defined ranges
                            for feature, (low, high) in NUMERICAL_FEATURES.items():
                                if isinstance(low, int) and isinstance(high, int):
                                    random_value = random.randint(low, high)
                                else:
                                    random_value = round(random.uniform(low, high), 2)
                                random_data[feature] = random_value
                            
                            # Convert random data to DataFrame
                            random_df = pd.DataFrame([random_data])
                            
                            # Append to application_features
                            st.session_state.application_features = pd.concat(
                                [st.session_state.application_features, random_df],
                                ignore_index=True
                            )
                            save_application_features(st.session_state.application_features)
                            app_features = random_data
                            st.success("Generated and saved new features for prediction.")
                        
                        # ---------------------- LABEL ENCODING ----------------------
                        # Create a copy to avoid modifying the original data
                        encoded_features = {}
                        for col in CATEGORICAL_FEATURES:
                            if col in label_encoders:
                                try:
                                    encoded_features[col] = label_encoders[col].transform([app_features[col]])[0]
                                except ValueError as e:
                                    st.error(f"Encoding Error in column '{col}': {e}")
                                    st.stop()
                            else:
                                encoded_features[col] = 0  # Default value or handle appropriately
                        
                        for col in NUMERICAL_FEATURES.keys():
                            encoded_features[col] = app_features[col]
                        
                        # Convert to DataFrame
                        encoded_df = pd.DataFrame([encoded_features])
                        
                        # ---------------------- FEATURE ALIGNMENT ----------------------
                        # Ensure the DataFrame has all required features
                        combined_data = encoded_df.reindex(columns=feature_names, fill_value=0)
                        
                        # ---------------------- PREDICTION ----------------------
                        try:
                            prediction = model.predict(combined_data)
                        except Exception as e:
                            st.error(f"Prediction Error: {e}")
                            st.stop()
                        
                        # ---------------------- DECODE PREDICTION ----------------------
                        try:
                            # Assuming 'Loan_Decision' was label encoded during training
                            loan_result = label_encoders["Loan_Decision"].inverse_transform(prediction)
                            suggested_status = loan_result[0]
                        except KeyError:
                            # If 'Loan_Decision' encoder is not available
                            suggested_status = prediction[0]
                        except Exception as e:
                            st.error(f"Decoding Error: {e}")
                            st.stop()
                        
                        st.write(f"**Model Prediction:** {suggested_status}")

                # 2. View/Generate PDF Report
                if st.button("View / Generate Report", key="report_btn"):
                    # Retrieve feature data
                    features_df = st.session_state.application_features
                    if selected_app_id in features_df["Application_ID"].values:
                        feature_row = features_df[features_df["Application_ID"] == selected_app_id].iloc[0].to_dict()
                    else:
                        st.warning("No features found for this application. Please generate prediction first.")
                        st.stop()
                    
                    html_report = generate_report_html(app_row.to_dict(), feature_row)

                    # Show the HTML inline
                    with st.expander("View Loan Report (HTML Preview)", expanded=True):
                        st.components.v1.html(html_report, height=700, scrolling=True)

                    # Convert HTML to PDF with WeasyPrint
                    pdf_bytes = html_to_pdf(html_report)
                    if pdf_bytes:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=f"LoanReport_{selected_app_id}.pdf",
                            mime="application/pdf"
                        )

                st.markdown("---")
                st.write("### Update Application Status")
                possible_statuses = ["Pending", "Approved", "Denied"]
                current_status = app_row["Status"] if app_row["Status"] in possible_statuses else "Pending"
                final_status = st.selectbox(
                    "Set Final Status", 
                    possible_statuses,
                    index=possible_statuses.index(current_status) 
                        if current_status in possible_statuses else 0,
                    key=f"status_select_{selected_app_id}"
                )
                if st.button("Confirm Status", key="confirm_status_btn"):
                    st.session_state.applications.loc[
                        st.session_state.applications["Application_ID"] == selected_app_id,
                        "Status"
                    ] = final_status
                    save_applications(st.session_state.applications)
                    st.success(f"Application {selected_app_id} updated to '{final_status}'")

else:
    st.warning("Please log in to access the system.")
