**Liver Cirrhosis Prediction Using Machine Learning**

**Project Overview:-**

Liver cirrhosis is a chronic and progressive liver disease that causes permanent damage to liver tissues and may lead to life-threatening complications if not diagnosed early. Conventional diagnostic approaches rely on multiple medical tests and expert judgment, which can delay early intervention.
This project implements a machine learning–based predictive system that analyzes patient clinical data to identify the presence of liver cirrhosis. The goal is to support healthcare professionals with accurate, early, and data-driven predictions.

Objectives:-
- Predict liver cirrhosis using supervised machine learning algorithms
- Compare the performance of multiple ML models
- Identify key clinical features influencing disease prediction
- Assist in early diagnosis and medical decision-making

Dataset Description:-
1) The dataset consists of patient medical records containing features such as:
2) Age
3) Gender
4) Total and direct bilirubin
5) Albumin
6) Prothrombin time
7) Liver enzymes (SGOT, SGPT, etc.)
8) Other relevant clinical parameters

Methodology:-
1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling and normalization

2. Exploratory Data Analysis (EDA):-
- Statistical analysis and visualization
- Feature correlation analysis
  
3. Machine Learning Models Used:-
The project implements and compares the following five machine learning algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

4. Model Evaluation:-
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

 Streamlit Dashboard:-
An interactive Streamlit web dashboard was developed to make the system user-friendly and accessible.

🔹 Dashboard Features
- User input form for patient clinical parameters
- Real-time liver cirrhosis prediction
- Model selection (Logistic Regression, SVM, KNN, Decision Tree, Random Forest)
- Display of prediction results
- Visualization of model performance metrics
The dashboard allows users to test different models and instantly view prediction outcomes.

Technologies Used:-
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Jupyter Notebook

Conclusion:-
This project demonstrates how machine learning combined with an interactive Streamlit dashboard can effectively predict liver cirrhosis using clinical data. The system enhances usability, supports early diagnosis, and improves healthcare decision-making.
