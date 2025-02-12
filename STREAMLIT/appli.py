import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import streamlit as st

# Fonction pour convertir les temps en secondes
def convert_time_to_seconds(time_str):
    try:
        if pd.isna(time_str):
            return None
        minutes, seconds = time_str.split(':')
        total_seconds = int(minutes) * 60 + float(seconds.replace(',', '.'))
        return total_seconds
    except (ValueError, AttributeError):
        return None

# Chargement et prétraitement des données
def load_and_preprocess_data(file_path):
    try:
        # Charger les données
        data = pd.read_csv(file_path, sep=';', parse_dates=['DOB'])
        print(f"Nombre initial de lignes : {len(data)}")

        # Supprimer les lignes avec des valeurs manquantes importantes
        data.dropna(subset=['Distance (in meters)', 'Stroke'], inplace=True)
        print(f"Lignes après suppression des valeurs manquantes : {len(data)}")

        # Filtrer les formats valides dans Results
        data = data[data['Results'].notna() & data['Results'].str.contains(r'^\d+:\d+,\d+$', na=False)]
        print(f"Lignes après filtrage des formats invalides dans Results : {len(data)}")

        # Convertir Results en secondes
        data['Results'] = data['Results'].apply(convert_time_to_seconds)
        data = data.dropna(subset=['Results'])
        print(f"Lignes après conversion des temps : {len(data)}")

        # Convertir certaines colonnes en types appropriés
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data = data.dropna(subset=['Year'])  # Supprimer les lignes avec des années non valides
        print(f"Lignes après validation de l'année : {len(data)}")

        # Filtrer par année
        data = data[(data['Year'] >= 1900) & (data['Year'] <= 2024)]
        print(f"Lignes après filtrage par année : {len(data)}")

        # Nettoyer et convertir Distance (in meters) et IMC en float
        data['Distance (in meters)'] = pd.to_numeric(data['Distance (in meters)'], errors='coerce')
        data['Height'] = pd.to_numeric(data['Height'], errors='coerce').fillna(data['Height'].mean())
        data['Weight'] = pd.to_numeric(data['Weight'], errors='coerce').fillna(data['Weight'].mean())

        # Calculer l'IMC si nécessaire
        data['IMC'] = data.apply(lambda row: row['Weight'] / ((row['Height'] / 100) ** 2) if row['Height'] > 0 else None, axis=1)

        # Normaliser l'année
        data['Year_normalized'] = data['Year'] - 1900

        # Ajouter la colonne Distance_IMC
        data['Distance_IMC'] = data['Distance (in meters)'] * data['IMC']
        data.dropna(subset=['Distance_IMC'], inplace=True)
        print(f"Lignes après calcul de Distance_IMC : {len(data)}")

        return data
    except Exception as e:
        print(f"Erreur lors du prétraitement des données : {e}")
        return pd.DataFrame()

# Entraîner le modèle de régression linéaire
def train_linear_model(data):
    if len(data) == 0:
        print("Aucune donnée disponible pour entraîner le modèle.")
        return None, None

    features = [
        'Year', 
        'Distance (in meters)', 
        'Stroke', 
        'Relay?', 
        'Age', 
        'IMC',
        'Year_normalized',
        'Distance_IMC'
    ]
    target = 'Results'

    # Encoder les variables catégorielles
    data_encoded = pd.get_dummies(data[features], drop_first=True)

    if len(data_encoded) < 5:
        print("Trop peu de données pour diviser en train/test. Entraînement sur tout le dataset.")
        model = LinearRegression()
        model.fit(data_encoded, data[target])
        mae = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            data_encoded, data[target], test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error: {mae}")

    # Sauvegarder le modèle et les colonnes
    joblib.dump(model, 'linear_regression_model.joblib')
    joblib.dump(list(data_encoded.columns), 'columns.joblib')

    return model, data_encoded.columns

# Faire des prédictions
def predict_future_results(model, columns, future_data):
    # Encoder les données futures
    future_data_encoded = pd.get_dummies(future_data, drop_first=True)

    # Assurer que les colonnes correspondent
    missing_cols = set(columns) - set(future_data_encoded.columns)
    extra_cols = set(future_data_encoded.columns) - set(columns)

    for col in missing_cols:
        future_data_encoded[col] = 0

    for col in extra_cols:
        future_data_encoded.drop(columns=[col], inplace=True)

    future_data_encoded = future_data_encoded[columns]

    # Faire les prédictions
    predictions = model.predict(future_data_encoded)
    future_data['Predicted Results'] = predictions

    return future_data[['Distance (in meters)', 'Stroke', 'Predicted Results']]

# Interface utilisateur Streamlit
def main():
    st.title("Prédiction des Performances des JO")
    file_path = r"C:\Users\OTMANE\Downloads\PROJET 3\NAGE_OLYMPIC_FINAL_V3.csv"
    data = load_and_preprocess_data(file_path)
    if len(data) == 0:
        st.error("Aucune donnée disponible après prétraitement. Veuillez vérifier le fichier CSV.")
        return

    try:
        model = joblib.load('linear_regression_model.joblib')
        columns = joblib.load('columns.joblib')
    except FileNotFoundError:
        model, columns = train_linear_model(data)
        if model is None:
            st.error("Impossible d'entraîner le modèle avec les données actuelles.")
            return

    year = st.number_input("Année", min_value=1900, max_value=2050, value=2028)
    distance = st.number_input("Distance (m)", min_value=50, max_value=1500, value=100)
    stroke = st.selectbox("Style", ['Freestyle', 'Backstroke', 'Breaststroke', 'Butterfly'])
    relay = st.selectbox("Relais ?", [0, 1])
    age = st.number_input("Âge", min_value=10, max_value=50, value=25)
    imc = st.number_input("IMC", min_value=15.0, max_value=35.0, value=22.0)

    input_data = pd.DataFrame({
        'Year': [year],
        'Distance (in meters)': [distance],
        'Stroke': [stroke],
        'Relay?': [relay],
        'Age': [age],
        'IMC': [imc],
        'Year_normalized': [year - 1900],
        'Distance_IMC': [distance * imc]
    })

    if st.button("Prédire"):
        prediction = predict_future_results(model, columns, input_data)
        st.success(f"Temps estimé : {prediction['Predicted Results'].values[0]:.2f} secondes")

if __name__ == "__main__":
    main()