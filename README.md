# Anveshak - Phishing Email Detection System

## 🎯 Project Overview

**Anveshak** is an intelligent phishing email detection system that uses machine learning to identify and flag suspicious emails. The project features a user-friendly web dashboard built with Streamlit that allows users to analyze email content in real-time.

## 🚀 Features

- **Real-time Email Analysis**: Paste raw email content and get instant phishing detection results
- **Machine Learning Model**: Trained on spam/ham datasets for accurate detection
- **Feature Extraction**: Analyzes multiple email characteristics including:
  - Body and subject length
  - Suspicious keyword detection
  - URL count analysis
  - Special character frequency
- **Confidence Scoring**: Provides confidence percentages for predictions
- **User-friendly Interface**: Clean, intuitive web dashboard

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation & Setup

### 1. Clone or Download the Project
```bash
# If you have the project files, navigate to the project directory
cd Anveshak-1
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Application
```bash
cd Conclave
python -m streamlit run Anveshak.py
```

### 4. Access the Dashboard
Once the application starts, it will automatically open in your default web browser at:
```
http://localhost:8501
```

## 🎮 How to Use

1. **Open the Dashboard**: Navigate to the URL shown in your terminal (usually http://localhost:8501)

2. **Paste Email Content**: Copy and paste the complete email content (including headers) into the text area

3. **Analyze**: Click the "Check Email" button to run the analysis

4. **View Results**: The system will display:
   - **Safe** (green) - Email appears legitimate
   - **Phishing** (red) - Email is likely a phishing attempt
   - Confidence percentage for the prediction

## 📁 Project Structure

```
Anveshak-1/
├── README.md
├── requirements.txt
└── Conclave/
    ├── Anveshak.py                    # Main Streamlit application
    ├── 20021010_easy_ham.tar.bz2      # Training dataset (legitimate emails)
    ├── 20021010_spam.tar.bz2          # Training dataset (spam emails)
    └── Phishing_Detection_model/
        ├── models/
        │   └── phishing_detector_v1.joblib  # Trained ML model
        ├── processed/
        │   └── email_features.csv            # Processed training data
        ├── Code/
        │   └── 1-Data_Cleaning.ipynb        # Data preprocessing notebook
        └── Model_Training_and_Evaluation.ipynb  # Model training notebook
```

## 🔧 Technical Details

### Machine Learning Model
- **Algorithm**: Scikit-learn based classifier
- **Features**: 5 engineered features from email content
- **Training Data**: SpamAssassin public datasets
- **Model Format**: Joblib serialized model

### Features Analyzed
1. **Body Length**: Character count in email body
2. **Subject Length**: Character count in email subject
3. **Suspicious Keywords**: Count of suspicious words (verify, update, login, urgent, etc.)
4. **URL Count**: Number of URLs in the email
5. **Special Characters**: Count of special characters (!$%^&*())

## 🛡️ Security Note

This tool is designed for educational and testing purposes. Always exercise caution when handling suspicious emails and never click on links or download attachments from untrusted sources.

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the machine learning model
- Enhancing the user interface

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: The dashboard should now be running! Check your terminal for the local URL and open it in your browser to start using the phishing detection system.