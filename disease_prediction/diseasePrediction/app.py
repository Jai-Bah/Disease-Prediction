import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import statistics

# Load and preprocess data (from Cells 1 and 2)
data = pd.read_csv("Training.csv").dropna(axis=1)
data = data.drop_duplicates()
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Define features and target (from Cell 3)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train models on full data (from Cell 6)
final_svm_model = SVC(probability=True, C=0.2)
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Create symptom index dictionary (from Cell 7)
symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Prediction function (from Cell 7)
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            st.warning(f"Symptom '{symptom}' not found in symptom index.")
    input_data = np.array(input_data).reshape(1, -1)
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Streamlit UI
st.title("Disease Prediction System")
st.write("Select symptoms to predict the likely disease.")

# Multi-select dropdown for symptoms
selected_symptoms = st.multiselect(
    "Select Symptoms",
    options=list(symptom_index.keys()),
    default=["Itching", "Skin Rash", "Nodal Skin Eruptions"]
)

# Predict button
if st.button("Predict Disease"):
    if selected_symptoms:
        symptoms_input = ",".join(selected_symptoms)
        predictions = predictDisease(symptoms_input)
        st.subheader("Prediction Results")
        st.write(f"**Random Forest Prediction**: {predictions['rf_model_prediction']}")
        st.write(f"**Naive Bayes Prediction**: {predictions['naive_bayes_prediction']}")
        st.write(f"**SVM Prediction**: {predictions['svm_model_prediction']}")
        st.write(f"**Final Prediction**: {predictions['final_prediction']}")
    else:
        st.error("Please select at least one symptom.")