import streamlit as st 
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import base64

# 📌 Fonction pour ajouter une image de fond avec une largeur réduite et bien centrée
def add_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{encoded_image}) no-repeat center center fixed;
            background-size: 50%;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 📌 Charger les données
@st.cache_data
def load_data():
    file_path = "C:/Users/flori/Desktop/PROJET_3/STREAMLIT/BDD/NAGE_OLYMPIC_FINAL_V3_MODIF_BI.csv"
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')

    # 🛠 Nettoyage des données
    df['Distance'] = df['Distance'].astype(str).str.strip()
    df['Stroke'] = df['Stroke'].str.strip().replace({
        'Freerelay Relay': 'Freestyle Relay',
        'Meelay Relay': 'Medley Relay',
        'Medlay Relay': 'Medley Relay'
    })
    
    # Conversion des résultats en secondes
    def convert_to_seconds(result):
        try:
            minutes, seconds = result.split(':')
            seconds, centièmes = seconds.split(',')
            return int(minutes) * 60 + int(seconds) + int(centièmes) / 100
        except:
            return None

    df['Results_sec'] = df['Results'].apply(convert_to_seconds)
    
    return df

# 📌 Appliquer l’image de fond
add_background(r"C:\Users\flori\Desktop\PROJET_3\STREAMLIT\images\swimmer.jpg")

# 📌 Interface : Menu de navigation
page = st.sidebar.selectbox("📌 Navigation", ["🏠 Accueil", "📊 Analyse des performances"], key="navigation_selectbox")

# 🎯 Page d'accueil
if page == "🏠 Accueil":
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Bienvenue sur le site des JO 2028</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center; color: white;'>Explorez les performances et les prévisions des épreuves de natation.</h3>",
        unsafe_allow_html=True
    )

# 🎯 Page d'analyse des performances
elif page == "📊 Analyse des performances":
    df = load_data()
    st.title("🏊 Prédiction des Temps aux JO 2028")

    # 🔽 Sélections des filtres
    st.sidebar.title("🎛️ Filtrer les Données")
    distance_selected = st.sidebar.selectbox("📏 Sélectionnez la distance :", ["Tout"] + sorted(df["Distance"].unique()), key="distance_selectbox")
    stroke_selected = st.sidebar.selectbox("🌊 Sélectionnez le style de nage :", ["Tout"] + sorted(df["Stroke"].unique()), key="stroke_selectbox")
    gender_selected = st.sidebar.selectbox("🚻 Sélectionnez le sexe :", ["Tout", "Men", "Women"], key="gender_selectbox")
    finalists_number = st.sidebar.selectbox("🏅 Sélectionnez le nombre de finalistes à afficher :", [1, 3, 8], key="finalists_selectbox")

    # 📊 Filtrer les données
    df_filtered = df
    if distance_selected != "Tout":
        df_filtered = df_filtered[df_filtered["Distance"] == str(distance_selected)]
    if stroke_selected != "Tout":
        df_filtered = df_filtered[df_filtered["Stroke"] == stroke_selected]
    if gender_selected != "Tout":
        df_filtered = df_filtered[df_filtered["Gender"] == gender_selected]

    # ✅ Vérification des données après filtrage
    if df_filtered.empty:
        st.warning("🚨 Aucune donnée disponible avec les critères sélectionnés.")
    else:
        # 🔄 Conversion du temps en secondes et préparation pour Prophet
        df_filtered = df_filtered[["Année", "Results_sec"]].dropna()
        df_filtered = df_filtered.rename(columns={"Année": "ds", "Results_sec": "y"})
        df_filtered["ds"] = pd.to_datetime(df_filtered["ds"], format="%Y")

        # 📈 Modèle Prophet pour la prédiction
        model = Prophet()
        model.fit(df_filtered)

        # 🔮 Faire une prédiction jusqu'en 2028
        future = model.make_future_dataframe(periods=5, freq='Y')
        forecast = model.predict(future)

        # 📌 Disposition en colonnes (2/3 pour le graphique, 1/3 pour les résultats)
        col1, col2 = st.columns([2, 1])

        # 📊 Colonne 1 : Affichage du graphique
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_alpha(0)  # Fond transparent
            ax.set_facecolor("none")  

            ax.plot(df_filtered["ds"], df_filtered["y"], "o-", color="white", linewidth=3, markersize=10, label="Historique")  
            ax.plot(forecast["ds"], forecast["yhat"], "--", color="black", linewidth=3, label="Prédiction")  

            ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="#FFD700", alpha=0.2)

            ax.set_xlabel("Année", color="white", fontsize=20, fontweight='bold')  
            ax.set_ylabel("Temps en secondes", color="white", fontsize=20, fontweight='bold')  
            ax.tick_params(axis='x', colors="white", labelsize=18)  
            ax.tick_params(axis='y', colors="white", labelsize=18)  
            ax.spines["top"].set_color("white")  
            ax.spines["right"].set_color("white")  
            ax.spines["left"].set_color("white")  
            ax.spines["bottom"].set_color("white")  
            ax.legend(facecolor="none", labelcolor="white", fontsize=16, loc='best')  

            st.pyplot(fig)

        # 📌 Colonne 2 : Affichage des résultats à droite
        with col2:
            forecast_2028 = forecast[forecast["ds"].dt.year == 2028]
            
            if not forecast_2028.empty:
                predicted_time_2028 = forecast_2028["yhat"].values[0]

                st.markdown("### ⏱ Prédictions des résultats aux JO 2028")
                
                if finalists_number == 1:
                    st.write(f"🥇 **1er finaliste : {predicted_time_2028:.2f} sec**")
                elif finalists_number == 3:
                    st.write(f"🥇 **1er finaliste : {predicted_time_2028:.2f} sec**")
                    st.write(f"🥈 **2e finaliste : {predicted_time_2028 + 0.5:.2f} sec**")
                    st.write(f"🥉 **3e finaliste : {predicted_time_2028 + 1.0:.2f} sec**")
                elif finalists_number == 8:
                    st.write(f"🥇 **1er finaliste : {predicted_time_2028:.2f} sec**")
                    for i in range(2, 9):
                        st.write(f"🏊‍♂️ **{i}e finaliste : {predicted_time_2028 + (i-1)*0.3:.2f} sec**")
            else:
                st.warning("❌ Aucune prédiction disponible pour 2028.")
