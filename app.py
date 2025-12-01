import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------
# Function to train & save models
# ---------------------------
def train_and_save_models():
    df = pd.read_csv("liver.dataset.csv")

    # Features & Target
    X = df[["Age", "total_Bilirubin", "Albumin", "total_Protiens"]]
    y = df["Dataset"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        filename = f"{name.replace(' ', '_')}.pkl"
        joblib.dump(model, filename)

# ---------------------------
# Train if .pkl files missing
# ---------------------------
if not all(os.path.exists(f"{m}.pkl") for m in
           ["Logistic_Regression","Decision_Tree","Random_Forest","SVM","KNN"]):
    st.warning("⚠️ Models not found! Training models now...")
    train_and_save_models()
    st.success("✅ Models trained and saved!")

# ---------------------------
# Streamlit Dashboard
# ---------------------------
st.set_page_config(page_title="Liver Cirrhosis Prediction", page_icon="🩺", layout="wide")

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50;'>🩺 Liver Cirrhosis Prediction Dashboard</h1>
    <p style='text-align: center; color: #7F8C8D;'>A Machine Learning Based Health Risk Prediction Tool</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Sidebar - Model choice
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]
)

# Load selected model
model = joblib.load(f"{model_choice.replace(' ', '_')}.pkl")

# Layout: Two Columns
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("👤 Enter Patient Details")

    age = st.slider("Age", 20, 80, 40)
    bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0, 1.0)
    albumin = st.number_input("Albumin", 1.0, 6.0, 3.5)
    proteins = st.number_input("Total Protiens", 4.0, 10.0, 6.5)

    features = np.array([[age, bilirubin, albumin, proteins]])

    if st.button("🔍 Predict", use_container_width=True):
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        with col2:
            st.subheader("📊 Prediction Results")
            if prediction == 1:
                st.error(f"⚠️ High Risk of Liver Disease\n\n**Probability:** {probability:.2f}%")
            else:
                st.success(f"✅ No Liver Disease Detected\n\n**Probability:** {100 - probability:.2f}%")

# Info Box
st.markdown("---")
st.info(f"ℹ️ Currently using **{model_choice}** for prediction. You can switch models from the sidebar.")