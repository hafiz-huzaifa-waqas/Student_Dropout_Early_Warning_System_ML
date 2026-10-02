import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# Load saved model, scaler, and training columns
# ------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    trained_columns = pickle.load(f)

# ------------------------------
# Risk labeling function
# ------------------------------
def risk_label(score, high_thresh=0.7, medium_thresh=0.4):
    if score >= high_thresh:
        return "High"
    elif score >= medium_thresh:
        return "Medium"
    else:
        return "Low"

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Student Dropout Early Warning System")
st.write("Upload student data to predict dropout risk and view high-risk students.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ------------------------------
    # Feature engineering
    # ------------------------------
    df['activity_score'] = df['raisedhands'] + df['VisITedResources'] + df['AnnouncementsView'] + df['Discussion']
    df['parent_engaged'] = df['ParentAnsweringSurvey'].map({'Yes':1, 'No':0})

    # One-hot encode categorical columns
    columns_to_encode = ['gender','NationalITy','PlaceofBirth','StageID','GradeID','SectionID',
                         'Topic','Semester','Relation','ParentAnsweringSurvey',
                         'ParentschoolSatisfaction','StudentAbsenceDays']

    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

    # Drop columns not used in training
    drop_columns = ['PlaceofBirth','Relation','SectionID','Class']
    X = df_encoded.drop(columns=drop_columns, errors='ignore')

    # Align columns with training data
    for col in trained_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_columns]  # reorder

    # ------------------------------
    # Optional scaling for LR/RF
    # ------------------------------
    model_name = type(model).__name__
    if model_name in ['LogisticRegression', 'RandomForestClassifier']:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values  # XGB can take raw features

    # ------------------------------
    # Risk thresholds sliders
    # ------------------------------
    st.sidebar.subheader("Set Risk Thresholds")
    high_thresh = st.sidebar.slider("High-risk threshold", 0.5, 1.0, 0.7, 0.01)
    medium_thresh = st.sidebar.slider("Medium-risk threshold", 0.0, 0.7, 0.4, 0.01)

    # ------------------------------
    # Predict risk
    # ------------------------------
    risk_scores = model.predict_proba(X_scaled)[:, 1]
    risk_labels = [risk_label(s, high_thresh, medium_thresh) for s in risk_scores]

    df_results = df.copy()
    df_results['risk_score'] = risk_scores
    df_results['risk_label'] = risk_labels
    df_results['predicted_dropout'] = (risk_scores >= 0.5).astype(int)

    # ------------------------------
    # Display Top 20 high-risk students
    # ------------------------------
    st.subheader("🏆 Top 20 High-Risk Students")
    top20 = df_results.sort_values('risk_score', ascending=False).head(20)
    st.dataframe(top20)
    st.bar_chart(top20.set_index(top20.index)['risk_score'])

    # ------------------------------
    # Select student to view details
    # ------------------------------
    st.subheader("🔍 View Individual Student Risk")
    student_id = st.selectbox("Select Student Index", df_results.index)
    student = df_results.loc[student_id]

    st.write(f"**Student Index:** {student_id}")
    st.write(f"**Risk Score:** {student['risk_score']:.2f}")
    st.write(f"**Risk Label:** {student['risk_label']}")
    st.write(f"**Predicted Dropout:** {student['predicted_dropout']}")

    # ------------------------------
    # Top contributing features for that student
    # ------------------------------
    feature_cols = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 
                    'activity_score', 'parent_engaged']
    feature_values = student[feature_cols].values
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Value': feature_values
    }).sort_values('Value', ascending=False).head(5)

    st.subheader("⚡ Top Reasons for Risk")
    st.table(feature_importance)

    # ------------------------------
    # Download CSV
    # ------------------------------
    st.subheader("💾 Download Predictions")
    st.download_button(
        label="Download Predictions CSV",
        data=df_results.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
