import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from datetime import datetime
import os

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
CATEGORICAL_FEATURES = [
    "Social_Media_Activity",
    "Utility_Payment_Timeliness",
    "Geolocation_Stability",
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

NUMERICAL_FEATURES = {
    "Online_Shopping_Frequency": [0, 10],
    "Mobile_Usage_Hours": [0, 12],
    "App_Subscriptions": [0, 20],
    "Streaming_Subscriptions": [0, 5],
    "Ecommerce_Transactions": [0, 15],
    "Ride_Sharing_Usage": [0, 30],
    "Smart_Device_Usage_Hours": [0, 24],
    "Digital_Payment_Transactions": [0, 50],
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

IMPORTANT_APPLICANT_FIELDS = [
    "Name", "Age", "Income", "Employment_Status",
    "Loan_Amount", "Loan_Purpose", "Status"
]

IMPORTANT_GENERATED_FEATURES = [
    "Social_Media_Activity",
    "Utility_Payment_Timeliness",
    "Geolocation_Stability",
    "Chist",
    "Cpur",
    "JobType",
    "creditScore"
]

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
        # For other features, assign default or 'unknown' values
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
    Adds an 'unknown' category to handle unseen data.
    """
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('unknown')  # Handle NaN values
        # Check if 'unknown' is already a category
        if 'unknown' not in df[col].unique():
            df[col] = df[col].astype(str) + '_unknown'
        le.fit(df[col])
        df[col] = le.transform(df[col])
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

def encode_application_data(app_features, label_encoders):
    """
    Encode application data using the provided label encoders.
    Handles unknown categories by assigning them to 'unknown'.
    """
    encoded_features = {}
    for col in CATEGORICAL_FEATURES:
        value = app_features.get(col, 'unknown')
        if pd.isna(value) or value == '':
            value = 'unknown'
        value = str(value).lower().strip()
        if value not in label_encoders[col].classes_:
            # Assign to 'unknown'
            if 'unknown' in label_encoders[col].classes_:
                value = 'unknown'
            else:
                # If 'unknown' not in classes, add it
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                value = 'unknown'
        encoded_features[col] = label_encoders[col].transform([value])[0]
    
    for col in NUMERICAL_FEATURES.keys():
        encoded_features[col] = app_features.get(col, 0)
    
    return pd.DataFrame([encoded_features])

def delete_application(application_id, username=None, user_type="customer"):
    """
    Delete an application from 'applications.csv' and 'application_features.csv'.
    For customers, ensure they can only delete their own applications.
    For bank admins, they can delete any application.
    """
    # Load applications
    if os.path.exists("applications.csv"):
        try:
            apps_df = pd.read_csv("applications.csv", dtype={"Application_ID": str})
        except Exception as e:
            st.error(f"Error loading 'applications.csv': {e}")
            return False
    else:
        st.error("'applications.csv' does not exist.")
        return False
    
    # Check if application exists
    if application_id not in apps_df["Application_ID"].values:
        st.error(f"Application ID {application_id} not found.")
        return False
    
    # For customers, verify ownership
    if user_type == "customer":
        app_row = apps_df[apps_df["Application_ID"] == application_id].iloc[0]
        if app_row["Username"] != username:
            st.error("You can only delete your own applications.")
            return False
    
    # Confirm deletion
    confirm = st.checkbox(f"Are you sure you want to delete Application ID {application_id}? This action cannot be undone.")
    if confirm:
        # Delete from applications.csv
        apps_df = apps_df[apps_df["Application_ID"] != application_id]
        try:
            apps_df.to_csv("applications.csv", index=False)
        except Exception as e:
            st.error(f"Error saving 'applications.csv': {e}")
            return False
        
        # Delete from application_features.csv
        if os.path.exists("application_features.csv"):
            try:
                features_df = pd.read_csv("application_features.csv", dtype={"Application_ID": str})
                features_df = features_df[features_df["Application_ID"] != application_id]
                features_df.to_csv("application_features.csv", index=False)
            except Exception as e:
                st.error(f"Error updating 'application_features.csv': {e}")
                return False
        else:
            st.warning("'application_features.csv' does not exist.")
        
        # Update session state
        if "applications" in st.session_state:
            st.session_state.applications = apps_df
        
        if "application_features" in st.session_state:
            st.session_state.application_features = features_df
        
        st.success(f"Application ID {application_id} has been successfully deleted.")
        return True
    else:
        st.info("Deletion not confirmed.")
        return False

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

                st.markdown("### Delete an Application")
                app_to_delete = st.selectbox(
                    "Select Application ID to Delete",
                    user_apps["Application_ID"].unique()
                )
                delete_button = st.button("Delete Selected Application", key="delete_customer")

                if delete_button:
                    delete_success = delete_application(
                        application_id=app_to_delete,
                        username=st.session_state.username,
                        user_type="customer"
                    )
                    if delete_success:
                        st.experimental_rerun()  # Refresh the app to reflect changes
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
                                    # Select a random category from known classes excluding 'unknown'
                                    possible_categories = [cls for cls in label_encoders[feature].classes_ if cls != 'unknown']
                                    if possible_categories:
                                        random_category = random.choice(possible_categories)
                                    else:
                                        random_category = 'unknown'
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
                        encoded_df = encode_application_data(app_features, label_encoders)
                        
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

                # 2. View Report
                if st.button("View Report", key="report_btn"):
                    # Retrieve feature data
                    features_df = st.session_state.application_features
                    if selected_app_id in features_df["Application_ID"].values:
                        feature_row = features_df[features_df["Application_ID"] == selected_app_id].iloc[0].to_dict()
                    else:
                        st.warning("No features found for this application. Please generate prediction first.")
                        st.stop()
                    
                    # Define the report data
                    report_data = {
                        "Applicant Information": {
                            "Name": app_row['Name'],
                            "Age": app_row['Age'],
                            "Monthly Income": f"${app_row['Income']}",
                            "Employment Status": app_row['Employment_Status'],
                            "Loan Amount Requested": f"${app_row['Loan_Amount']}",
                            "Loan Purpose": app_row['Loan_Purpose'],
                            "Current Status": app_row['Status']
                        },
                        "Generated Features": {},
                        "Model Prediction": "Pending"
                    }

                    # Populate Generated Features
                    for feature in IMPORTANT_GENERATED_FEATURES:
                        value = feature_row.get(feature, "N/A")
                        if feature in label_encoders:
                            try:
                                decoded_value = label_encoders[feature].inverse_transform([value])[0]
                                display_value = decoded_value.replace("_", " ").capitalize()
                            except:
                                display_value = value
                        else:
                            display_value = value
                        # Replace 'unknown_unknown' with 'Unknown'
                        if display_value == 'unknown_unknown':
                            display_value = 'Unknown'
                        report_data["Generated Features"][feature.replace('_', ' ')] = display_value

                    # Populate Model Prediction
                    if "Loan_Decision" in label_encoders and "Loan_Decision" in feature_row:
                        try:
                            prediction_encoded = feature_row["Loan_Decision"]
                            prediction = label_encoders["Loan_Decision"].inverse_transform([prediction_encoded])[0]
                            report_data["Model Prediction"] = prediction
                        except:
                            report_data["Model Prediction"] = "Error in prediction"
                    else:
                        report_data["Model Prediction"] = "Pending"

                    # ---------------------- DISPLAY REPORT ----------------------
                    st.markdown("### 📝 Loan Application Report")
                    
                    # Applicant Information
                    with st.container():
                        st.markdown("#### 📄 Applicant Information")
                        cols_applicant = st.columns(2)
                        for idx, (key, value) in enumerate(report_data["Applicant Information"].items()):
                            with cols_applicant[idx % 2]:
                                st.write(f"**{key}:** {value}")

                    # Generated Features
                    with st.container():
                        st.markdown("#### 🔍 Generated Features")
                        cols_features = st.columns(2)
                        for idx, (key, value) in enumerate(report_data["Generated Features"].items()):
                            with cols_features[idx % 2]:
                                st.write(f"**{key}:** {value}")

                    # Model Prediction
                    with st.container():
                        st.markdown("#### 🤖 Model Prediction")
                        if report_data["Model Prediction"] != "Pending":
                            st.success(f"**{report_data['Model Prediction']}**")
                        else:
                            st.warning("Model prediction is pending.")

                    # ---------------------- DOWNLOAD REPORT ----------------------
                    # Prepare report data for download
                    # Flatten the report data
                    flat_report = {}
                    for section, details in report_data.items():
                        if isinstance(details, dict):
                            for key, value in details.items():
                                flat_report[f"{section} - {key}"] = value
                        else:
                            flat_report[section] = details

                    # Convert to DataFrame for CSV
                    report_df = pd.DataFrame([flat_report])

                    # Download as CSV
                    st.download_button(
                        label="📥 Download Report as CSV",
                        data=report_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"LoanReport_{selected_app_id}.csv",
                        mime="text/csv"
                    )

                    # Download as TXT
                    report_text = "Loan Application Report\n\n"
                    for section, details in report_data.items():
                        if isinstance(details, dict):
                            report_text += f"{section}:\n"
                            for key, value in details.items():
                                report_text += f"  - {key}: {value}\n"
                        else:
                            report_text += f"{section}: {details}\n"
                        report_text += "\n"
                    st.download_button(
                        label="📥 Download Report as TXT",
                        data=report_text.encode('utf-8'),
                        file_name=f"LoanReport_{selected_app_id}.txt",
                        mime="text/plain"
                    )

                # 3. Delete Application
                st.markdown("### Delete an Application")
                delete_button_bank = st.button("Delete Selected Application", key="delete_bank")

                if delete_button_bank:
                    delete_success = delete_application(
                        application_id=selected_app_id,
                        user_type="bank"
                    )
                    if delete_success:
                        st.experimental_rerun()  # Refresh the app to reflect changes

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
