import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Bienvenue sur mon API Flask"})

@app.route('/predictkaggle', methods=['GET'])
def predict():
    try:
        # Récupérer les hyperparamètres du modèle depuis les paramètres de la requête GET
        fit_intercept = request.args.get('fit_intercept', 'True') == 'True'

        # Charger les données (remplacer ce chemin par celui de votre propre fichier CSV)
        df = pd.read_csv("data/HomeC.csv", usecols=['temperature', 'time'])

        # Vérifier les premières lignes du DataFrame
        print("Données brutes :")
        print(df.head())

        # Nettoyer les données en forçant les valeurs à être numériques (remplacer les non-numeriques par NaN)
        df['time'] = pd.to_numeric(df['time'], errors='coerce')  # Convertir en numérique, en forçant les erreurs à NaN

        # Retirer les lignes contenant des valeurs NaN dans 'time' ou 'temperature'
        df.dropna(subset=['time', 'temperature'], inplace=True)

        # Convertir la colonne 'time' en datetime
        df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')  # Convertir les timestamps UNIX en datetime

        # Vérifier les premières lignes après la conversion
        print("Données après conversion de 'time' en datetime :")
        print(df.head())

        # Convertir la colonne 'time' en nombre de secondes depuis l'époque pour l'utilisation avec le modèle
        df['time'] = df['time'].astype('int64') / 10**9  # Conversion en secondes depuis l'époque (format float)

        # Calculer la différence de température entre les valeurs successives
        df['temp_diff'] = df['temperature'].diff().abs()

        # Filtrer les données pour ne garder que celles où la différence de température est significative
        df_filtered = df[df['temp_diff'] > 0.1]  # Seuil de 0.1°C pour la variation

        # Vérifier les données filtrées
        print("Données filtrées (différence de température > 0.1°C) :")
        print(df_filtered.head())

        # Normaliser les données de température
        scaler_temp = MinMaxScaler(feature_range=(0, 1))
        df_filtered['temperature_scaled'] = scaler_temp.fit_transform(df_filtered['temperature'].values.reshape(-1, 1))

        # Afficher les valeurs normalisées
        print("Données normalisées :")
        print(df_filtered[['temperature', 'temperature_scaled']].head())

        # Séparer les données en entrées (X) et sorties (y)
        X = df_filtered['time'].values.reshape(-1, 1)  # Utiliser 'time' comme variable d'entrée
        y = df_filtered['temperature_scaled'].values  # Utiliser 'temperature_scaled' comme variable de sortie normalisée

        # Créer et entraîner le modèle de régression linéaire avec les hyperparamètres spécifiés
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, y)

        # Prédire la température pour la prochaine valeur de 'time'
        last_time = df_filtered['time'].values[-1]
        next_time = last_time + (df_filtered['time'].values[1] - df_filtered['time'].values[0])  # Calculer le temps suivant
        prediction_scaled = model.predict(np.array([[next_time]]))
        
        # Dénormaliser la prédiction
        prediction = scaler_temp.inverse_transform(prediction_scaled.reshape(-1, 1))

        # Retourner la prédiction sous forme de JSON
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
