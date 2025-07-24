import streamlit as st
import pandas as pd
import joblib
import os
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import re

# --- Helper Functions (Copied from your notebooks) ---

def parse_email_from_text(raw_text):
    """Parses raw email text and extracts its content."""
    # The text needs to be encoded to bytes for the parser to work correctly
    msg = BytesParser(policy=policy.default).parse(raw_text.encode('utf-8'))
    
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            # Find the plain text part of the email, ignore attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
    else:
        # If not multipart, just get the main content
        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        
    # Use BeautifulSoup to remove any HTML tags
    soup = BeautifulSoup(body, 'html.parser')
    clean_body = soup.get_text()
    
    return {
        'subject': msg['subject'],
        'body': clean_body
    }

def extract_features(df):
    """Takes a DataFrame with email content and adds the feature columns your model needs."""
    df['body'] = df['body'].astype(str)
    df['subject'] = df['subject'].astype(str)
    
    # --- These features MUST be the same as in your training notebook ---
    df['body_length'] = df['body'].str.len()
    df['subject_length'] = df['subject'].str.len()
    
    suspicious_keywords = ['verify', 'update', 'login', 'urgent', 'account', 'password', 'suspend', 'confirm']
    df['suspicious_keyword_count'] = df['body'].apply(lambda x: sum(keyword in x.lower() for keyword in suspicious_keywords))
    
    df['url_count'] = df['body'].apply(lambda x: len(re.findall(r'http[s]?://', x)))
    
    df['special_char_count'] = df['body'].apply(lambda x: len(re.findall(r'[!$%^&*()]', x)))
    
    return df


# --- Load The Trained Model ---

# Use st.cache_resource to load the model only once, making the app faster.
@st.cache_resource
def load_model():
    """Loads the saved phishing detection model from the correct path."""
    # This path is relative to where you run streamlit (i.e., from /Users/om/Conclave)
    model_path = 'Phishing_Detection_model/models/phishing_detector_v1.joblib'
    model = joblib.load(model_path)
    return model

model = load_model()

# --- Streamlit Web Application Interface ---

st.set_page_config(page_title="Phishing Detector", page_icon="🎣")
st.title("Phishing Email Detector 🎣")
st.write("This tool uses a machine learning model to analyze email content and predict if it's a phishing attempt. Paste the full email content below to check.")

# Text area for user to paste email content
email_text = st.text_area("Email Content", height=300, placeholder="Paste raw email text here, including headers like 'Subject:' and 'From:'...")

# Button to trigger the analysis
if st.button("Check Email"):
    if email_text:
        # 1. Parse the raw email text using your helper function
        parsed_email = parse_email_from_text(email_text)
        
        # 2. Convert the parsed dictionary to a DataFrame (with one row)
        df = pd.DataFrame([parsed_email])
        
        # 3. Extract features using your helper function
        df_features = extract_features(df)
        
        # 4. Prepare features for the model. 
        # The columns must be in the exact same order as when you trained the model.
        # Drop the text columns and keep only the feature columns.
        feature_columns_in_order = ['body_length', 'subject_length', 'suspicious_keyword_count', 'url_count', 'special_char_count']
        features_for_model = df_features[feature_columns_in_order]
        
        # 5. Make a prediction using the loaded model
        prediction = model.predict(features_for_model)
        prediction_proba = model.predict_proba(features_for_model)
        
        # 6. Display the result
        st.write("---")
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.error(" This email is likely a **Phishing** attempt.")
            st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success(" This email appears to be **Safe**.")
            st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
    else:
        st.warning("Please paste some email content to check.")
