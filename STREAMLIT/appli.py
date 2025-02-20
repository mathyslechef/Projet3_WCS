import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import prophet
from prophet import prophet
from fbprophet import Prophet
import re
import base64

# Fonction pour encoder une image en base64 et l'appliquer comme fond d'écran
def add_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    background_image_style = f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{encoded_string});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1 {{
        color: white;
    }}
    </style>
    """
    st.markdown(background_image_style, unsafe_allow_html=True)

# Fonction pour charger et nettoyer les données
def load_and_clean_data(file_path):
    try:
        with st.spinner("Chargement des données..."):
            # Charger les données avec le séparateur ';'
            data = pd.read_csv(file_path, sep=';', engine='python', on_bad_lines='skip')

        # Traiter les colonnes numériques avec des virgules comme séparateurs décimaux
        numeric_columns = ['IMC', 'Results']
        for column in numeric_columns:
            if column in data.columns and data[column].dtype == 'object':
                data[column] = data[column].str.replace(',', '.')
                data[column] = pd.to_numeric(data[column], errors='coerce')

        # Supprimer les lignes avec des valeurs non convertibles
        data.dropna(subset=['IMC', 'Results'], inplace=True)

        # Calculer l'âge au moment de la compétition
        data['Age'] = data['Year'] - data['YOB']

        # Traiter les distances avec des unités comme '100m' ou '4x100'
        def process_distance(value):
            if isinstance(value, str):
                if 'x' in value:
                    parts = value.split('x')
                    return int(parts[0]) * int(parts[1])
                else:
                    return int(re.search(r'\d+', value).group())
            elif isinstance(value, (int, float)):
                return value
            else:
                return None

        data['Distance (in meters)'] = data['Distance (in meters)'].apply(process_distance)

        # Convertir les résultats en secondes (ex. '00:04:47.600000' -> 287.6)
        def results_to_seconds(result):
            if isinstance(result, str):
                try:
                    minutes, seconds = result.split(':')
                    return float(minutes) * 60 + float(seconds)
                except ValueError:
                    try:
                        return float(result)  # Si déjà au format numérique
                    except ValueError:
                        return None
            elif isinstance(result, (int, float)):
                return result
            else:
                return None

        data['Results'] = data['Results'].apply(results_to_seconds)
        data.dropna(subset=['Results'], inplace=True)

        # Encodage des variables catégorielles
        label_encoders = {}
        categorical_columns = ['Gender', 'Stroke', 'Relay?']
        for column in categorical_columns:
            if column in data.columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                label_encoders[column] = le

        # Normalisation des caractéristiques numériques restantes
        scaler = StandardScaler()
        numeric_features = ['IMC', 'Age', 'Distance (in meters)']
        existing_numeric_features = [f for f in numeric_features if f in data.columns]
        if existing_numeric_features:
            data[existing_numeric_features] = scaler.fit_transform(data[existing_numeric_features])

        return data, label_encoders, scaler, numeric_features

    except FileNotFoundError:
        st.error("Le fichier CSV n'a pas été trouvé. Vérifiez le chemin.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement et du nettoyage des données : {e}")
        st.stop()

# Fonction pour entraîner le modèle
def train_model(data, features, target):
    historical_data = data[data['Year'] <= 2024]

    # Assurer que toutes les colonnes nécessaires sont présentes
    for col in features:
        if col not in historical_data.columns:
            historical_data[col] = 0

    X = historical_data[features]
    y = historical_data[target]

    # Optimisation des hyperparamètres avec GridSearchCV
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(X, y)

    return final_model, best_params, best_score, X.columns

# Fonction pour faire des prédictions
def make_predictions(model, scaler, label_encoders, unique_events, historical_data, feature_columns):
    predictions_2028 = []
    for _, row in unique_events.iterrows():
        event_data = {
            'Year': 2028,
            'Distance (in meters)': row['Distance (in meters)'],
            'Stroke': encode_value(label_encoders, 'Stroke', row['Stroke']),
            'Relay?': encode_value(label_encoders, 'Relay?', row['Relay?']),
            'IMC': historical_data['IMC'].mean() if 'IMC' in historical_data.columns else 0,
            'Gender': 0,
            'Age': historical_data['Age'].mean() if 'Age' in historical_data.columns else 0
        }

        input_df = pd.DataFrame([event_data])

        # Assurer que toutes les colonnes nécessaires sont présentes
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Réorganiser strictement les colonnes
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Normaliser les caractéristiques numériques
        existing_numeric_features = [f for f in numeric_features if f in input_df.columns]
        if existing_numeric_features:
            input_df[existing_numeric_features] = scaler.transform(input_df[existing_numeric_features])

        prediction = model.predict(input_df)[0]
        predictions_2028.append({
            'Distance (in meters)': row['Distance (in meters)'],
            'Stroke': decode_value(label_encoders, 'Stroke', row['Stroke']),
            'Relay?': decode_value(label_encoders, 'Relay?', row['Relay?']),
            'Predicted Result': prediction
        })

    return pd.DataFrame(predictions_2028)

# Fonction pour encoder une valeur catégorielle
def encode_value(label_encoders, column, value):
    if column in label_encoders:
        try:
            return label_encoders[column].transform([str(value)])[0]
        except ValueError:
            return 0
    return 0

# Fonction pour décoder une valeur catégorielle
def decode_value(label_encoders, column, value):
    if column in label_encoders:
        try:
            return label_encoders[column].inverse_transform([int(value)])[0]
        except ValueError:
            return "Inconnu"
    return "Inconnu"

# Ajouter l'image en arrière-plan
add_background(r"C:\Users\OTMANE\Downloads\PROJET 3\p3.diapo2.jpg")

# Titre de l'application
st.title("Prédiction des Performances des Jeux Olympiques 2028")

# Chemin vers le fichier CSV
file_path = r"C:\Users\OTMANE\Downloads\PROJET 3\NAGE_OLYMPIC_FINAL.csv"

# Charger et nettoyer les données
data, label_encoders, scaler, numeric_features = load_and_clean_data(file_path)

# Définir les colonnes pertinentes pour la prédiction
features = ['Year', 'Distance (in meters)', 'Stroke', 'Relay?', 'IMC', 'Gender', 'Age']
target = 'Results'

# Entraîner le modèle
final_model, best_params, best_score, feature_columns = train_model(data, features, target)
st.write(f"Meilleurs paramètres : {best_params}")
st.write(f"Score RMSE moyen (validation croisée) : {best_score:.2f}")

# Évaluation du modèle avec validation croisée
cv_scores = cross_val_score(final_model, data[features], data[target], cv=5, scoring='neg_mean_squared_error')
rmse_scores = (-cv_scores) ** 0.5
st.write(f"RMSE moyen (validation croisée) : {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")

# Prédictions pour 2028
st.subheader("Prédictions pour les Jeux Olympiques de 2028")

# Forcer l'inclusion de toutes les distances
all_distances = [50, 100, 200, 400, 800, 1500]
unique_strokes = data['Stroke'].unique() if 'Stroke' in data.columns else [0]
unique_relays = data['Relay?'].unique() if 'Relay?' in data.columns else [0]

# Créer un DataFrame avec toutes les combinaisons possibles
unique_events = []
for distance in all_distances:
    for stroke in unique_strokes:
        for relay in unique_relays:
            unique_events.append({'Distance (in meters)': distance, 'Stroke': stroke, 'Relay?': relay})
unique_events = pd.DataFrame(unique_events)

predictions_df = make_predictions(final_model, scaler, label_encoders, unique_events, data, feature_columns)

# Ajouter "Tous" aux options de sélection
all_strokes = ['Tous'] + list(predictions_df['Stroke'].unique())
all_distances = ['Tous'] + [str(d) for d in sorted(predictions_df['Distance (in meters)'].unique())]
all_relays = ['Tous', 'Oui', 'Non']
all_genders = ['Tous', 'Homme', 'Femme']
top_results_options = ['Meilleur', 'Top 3', 'Top 5', 'Top 8']

# Menus déroulants dans la sidebar
with st.sidebar:
    selected_stroke = st.selectbox("Choisir une épreuve", all_strokes)
    selected_distance = st.selectbox("Choisir une distance", all_distances)
    selected_relay = st.selectbox("Choisir si c'est un relais", all_relays)
    selected_gender = st.selectbox("Choisir un genre", all_genders)
    selected_top_results = st.selectbox("Nombre de résultats à afficher", top_results_options)

# Filtre les résultats en fonction des sélections
filtered_df = predictions_df.copy()

if selected_stroke != 'Tous':
    filtered_df = filtered_df[filtered_df['Stroke'] == selected_stroke]

if selected_distance != 'Tous':
    filtered_df = filtered_df[filtered_df['Distance (in meters)'] == int(selected_distance)]

if selected_relay != 'Tous':
    relay_code = 1 if selected_relay == 'Oui' else 0
    filtered_df = filtered_df[filtered_df['Relay?'] == ('Oui' if relay_code == 1 else 'Non')]

if selected_gender != 'Tous':
    gender_code = 0 if selected_gender == 'Homme' else 1
    filtered_df = filtered_df[filtered_df['Stroke'].isin(
        data[data['Gender'] == gender_code]['Stroke'].unique()
    )]

filtered_df = filtered_df.sort_values(by='Predicted Result')

# Afficher le nombre de résultats choisi
if not filtered_df.empty:
    if selected_top_results == 'Meilleur':
        st.write(filtered_df.head(1))
    elif selected_top_results == 'Top 3':
        st.write(filtered_df.head(3))
    elif selected_top_results == 'Top 5':
        st.write(filtered_df.head(5))
    elif selected_top_results == 'Top 8':
        st.write(filtered_df.head(8))
else:
    st.write("Aucun résultat ne correspond à vos critères.")

# Mode personnalisé pour les prédictions
st.subheader("Prédiction Personnalisée")
custom_imc = st.number_input("Indice de Masse Corporelle (IMC)", min_value=10.0, max_value=50.0, value=22.0)
custom_age = st.number_input("Âge", min_value=10, max_value=100, value=25)

if st.button("Prédire ma Performance"):
    custom_event_data = {
        'Year': 2028,
        'Distance (in meters)': 100,  # Distance fixe pour la démonstration
        'Stroke': encode_value(label_encoders, 'Stroke', 'Nage libre'),
        'Relay?': encode_value(label_encoders, 'Relay?', 'Non'),
        'IMC': custom_imc,
        'Gender': 0,
        'Age': custom_age
    }

    custom_input_df = pd.DataFrame([custom_event_data])

    # Assurer que toutes les colonnes nécessaires sont présentes
    for col in feature_columns:
        if col not in custom_input_df.columns:
            custom_input_df[col] = 0

    # Réorganiser strictement les colonnes
    custom_input_df = custom_input_df.reindex(columns=feature_columns, fill_value=0)

    # Normaliser les caractéristiques numériques
    existing_numeric_features = [f for f in numeric_features if f in custom_input_df.columns]
    if existing_numeric_features:
        custom_input_df[existing_numeric_features] = scaler.transform(custom_input_df[existing_numeric_features])

    custom_prediction = final_model.predict(custom_input_df)[0]
    st.success(f"Votre performance prédite : {custom_prediction:.2f} secondes")
