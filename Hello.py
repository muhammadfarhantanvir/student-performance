import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data function
def load_data(file):
    df = pd.read_csv(file)
    return df

# Data preprocessing function
def preprocess_data(df):
    # Your existing data preprocessing code here
    return df

# Train model function
def train_model(X_train_scaled, y_train):
    model_rf = RandomForestRegressor(random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    return model_rf

# Main function
def main():
    st.title("Lernfortschritt Analysis")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load and preprocess data
        df = load_data(uploaded_file)
        df = preprocess_data(df)

        # Display the raw data
        st.subheader("Raw Data")
        st.write(df)
        # Annahme: Auswertung von Lernaufgaben basiert auf der durchschnittlichen Bewertung der Aufgaben
        df['Auswertung_Lernaufgaben'] = df[['Abgabe1', 'Abgabe2', 'Abgabe3']].mean(axis=1)

        # Annahme: Lernaktivitäten basieren auf der Summe verschiedener Aktivitäten
        df['Lernaktivitaeten'] = df[['Anz_Zugriffe', 'Anz_Forum', 'Anz_Post', 'Anz_Quiz_Pruefung']].sum(axis=1)
        # Split data and train model
        features = [
            'Auswertung_Lernaufgaben', 'Lernaktivitaeten',
            'Anz_Anmeldungen', 'Anz_Zugriffe', 'Anz_Forum', 'Anz_Post', 'Anz_Quiz_Pruefung',
            'Abgabe1_spaet', 'Abgabe2_spaet', 'Abgabe3_spaet',
            'Abgabe1_stunden', 'Abgabe2_stunden', 'Abgabe3_stunden',
            'Abgabe_mittel'
        ]
        target = 'Abschlussnote'

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train the model
        model_rf = train_model(X_train_scaled, y_train)

        # Overall Results
        st.subheader("Overall Results")
        y_pred = model_rf.predict(scaler.transform(X_test))
        mse_rf = mean_squared_error(y_test, y_pred)
        st.write(f'Mean Squared Error (Random Forest): {mse_rf}')

        # Filter for Each Student
        st.subheader("Filter for Each Student")
        student_list = df['Student_ID'].unique()
        selected_student = st.selectbox("Select a Student ID", student_list)

        # Filter data for the selected student
        filtered_data = df[df['Student_ID'] == selected_student]
        
        # Display filtered data
        st.write(filtered_data)

        # Predictions for the selected student
        X_selected = filtered_data[features]
        y_selected_pred = model_rf.predict(scaler.transform(X_selected))

        # Display predictions
        st.write(f'Predicted Abschlussnote for {selected_student}: {y_selected_pred}')

if __name__ == "__main__":
    main()
