import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for model saving/loading

# Load data function
def load_data(file):
    df = pd.read_csv(file)
    return df

# Data preprocessing function
def preprocess_data(df):
    # Annahme: Auswertung von Lernaufgaben basiert auf der durchschnittlichen Bewertung der Aufgaben
    df['Auswertung_Lernaufgaben'] = df[['Abgabe1', 'Abgabe2', 'Abgabe3']].mean(axis=1)

    # Annahme: Lernaktivitäten basieren auf der Summe verschiedener Aktivitäten
    df['Lernaktivitaeten'] = df[['Anz_Zugriffe', 'Anz_Forum', 'Anz_Post', 'Anz_Quiz_Pruefung']].sum(axis=1)

    return df

# Train model function
def train_model(X_train_scaled, y_train):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model

# Save model function
def save_model(model, filename):
    joblib.dump(model, filename)

# Load model function
def load_model(filename):
    return joblib.load(filename)

# Main function
def main():
    st.title("Lernfortschritt Analysis")

    # Option selection
    option = st.radio("Select Option", ["Train New Model", "Use Pretrained Model"])

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load and preprocess data
        df = load_data(uploaded_file)
        df = preprocess_data(df)

        # Display the raw data (with an option to hide/show)
        show_raw_data = st.checkbox("Show Raw Data", value=True)
        if show_raw_data:
            st.subheader("Raw Data")
            st.write(df)

        scaler = None  # Initialize scaler outside the if blocks
        X_test = None  # Initialize X_test outside the if blocks
        y_test = None  # Initialize y_test outside the if blocks

        if option == "Train New Model":
            # Feature selection (exclude string-type features)
            all_features = df.columns.tolist()
            features_to_remove = st.multiselect("Select Features to Remove", ['Student_ID'] + all_features, ['Student_ID'])

            # Remove selected features
            features = list(set(all_features) - set(features_to_remove))

            # Split data and train model
            target = 'Abschlussnote'
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Train the model
            model = train_model(X_train_scaled, y_train)

            # Save the model
            save_model(model, "linear_regression_model.joblib")

        elif option == "Use Pretrained Model":
            # Load the pretrained model
            model = load_model("linear_regression_model.joblib")

            # Feature selection for pretrained model
            numeric_features = df.select_dtypes(include=['number']).columns.tolist()
            features_to_use = st.multiselect("Select Features to Use", numeric_features, numeric_features)

            # Use selected features
            features = features_to_use

            # Initialize X_test and y_test for the pretrained model
            X_test, _ = train_test_split(df[features], test_size=0.2, random_state=42)
            y_test = df.loc[df.index.isin(X_test.index), 'Abschlussnote']


        # Overall Results
        st.subheader("Overall Results")
        X_test_scaled = scaler.transform(X_test) if scaler else X_test
        y_pred_test = model.predict(X_test_scaled)
        mse_test = mean_squared_error(y_test, y_pred_test)
        st.write(f'Mean Squared Error (Linear Regression): {mse_test}')

        if option == "Train New Model":
            # Filter for Each Student
            st.header("Filter for Each Student")
            student_list = df['Student_ID'].unique()
            selected_student = st.selectbox("Select a Student ID", student_list)

            # Filter data for the selected student
            filtered_data = df[df['Student_ID'] == selected_student]

            # Display filtered data
            st.subheader("Filtered Data for Selected Student")
            st.write(filtered_data)

            # Predictions for the selected student
            X_selected = filtered_data[features]
            X_selected_scaled = scaler.transform(X_selected) if scaler else X_selected
            y_selected_pred = model.predict(X_selected_scaled)

            # Display predictions
            st.subheader("Predicted performance for selected student")
            st.write(f'Predicted Abschlussnote for {selected_student}: {y_selected_pred}')

        elif option == "Use Pretrained Model":
            # Predictions for a specific student ID
            st.header("Predictions for a Specific Student ID")
            student_list = df['Student_ID'].unique()
            selected_student = st.selectbox("Select a Student ID", student_list)

            # Filter data for the selected student
            filtered_data = df[df['Student_ID'] == selected_student]

            # Display filtered data
            st.subheader("Filtered Data for Selected Student")
            st.write(filtered_data)

            # Predictions for the selected student using pretrained model
            X_selected = filtered_data[features]
            X_selected_scaled = scaler.transform(X_selected) if scaler else X_selected
            y_selected_pred = model.predict(X_selected_scaled)

            # Display predictions
            st.subheader("Predicted performance for selected student using pretrained model")
            st.write(f'Predicted Abschlussnote for {selected_student}: {y_selected_pred}')

if __name__ == "__main__":
    main()
