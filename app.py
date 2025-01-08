import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from flask_cors import CORS

app = Flask(__name__)

# Configurer CORS pour toutes les routes
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Bienvenue sur mon API Flask"})

@app.route('/predictkaggle', methods=['GET'])
def predict():
    # Charger les données (vous pouvez remplacer ce chemin par celui de votre propre fichier CSV)
    df = pd.read_csv("data/HomeC.csv", usecols=['temperature', 'time'])

    # Vérifier les premières lignes du DataFrame
    print(df.head())

    # Nettoyer les données en forçant les valeurs à être numériques (remplacer les non-numeriques par NaN)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')  # Convertir en numérique, en forçant les erreurs à NaN

    # Retirer les lignes contenant des valeurs NaN dans 'time' ou 'temperature'
    df.dropna(subset=['time', 'temperature'], inplace=True)

    # Convertir la colonne 'time' en datetime
    df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')  # Convertir les timestamps UNIX en datetime

    # Vérifier les premières lignes après la conversion
    print(df.head())

    # Convertir la colonne 'time' en nombre de secondes depuis l'époque pour l'utilisation avec le modèle
    df['time'] = df['time'].astype('int64') / 10**9  # Conversion en secondes depuis l'époque (format float)

    # Normaliser les données de température
    scaler = MinMaxScaler(feature_range=(0, 1))
    temps_scaled = scaler.fit_transform(df['temperature'].values.reshape(-1, 1))

    # Séparer les données en entrées (X) et sorties (y)
    X = df['time'].values.reshape(-1, 1)  # Utiliser 'time' comme variable d'entrée
    y = df['temperature'].values

    # Créer et entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)

    # Prédire la température pour la prochaine valeur de 'time'
    last_time = df['time'].values[-1]
    next_time = last_time + (df['time'].values[1] - df['time'].values[0])  # Calculer le temps suivant
    prediction = model.predict(np.array([[next_time]]))
    
    # Retourner la prédiction sous forme de JSON
    return jsonify({"prediction": prediction.tolist()})


# Nouvelle route pour tester le modèle k-Nearest Neighbors (k-NN)
@app.route('/predict_knn', methods=['GET'])
def predict_knn():
    try:
        # Récupérer les hyperparamètres du modèle k-NN depuis les paramètres de la requête GET
        n_neighbors = int(request.args.get('n_neighbors', 5))

        # Charger les données
        df = pd.read_csv("data/HomeC.csv", usecols=['temperature', 'time'])

        # Nettoyer les données en forçant les valeurs à être numériques (remplacer les non-numeriques par NaN)
        df['time'] = pd.to_numeric(df['time'], errors='coerce')  # Convertir en numérique, en forçant les erreurs à NaN

        # Retirer les lignes contenant des valeurs NaN dans 'time' ou 'temperature'
        df.dropna(subset=['time', 'temperature'], inplace=True)

        # Convertir la colonne 'time' en datetime
        df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')  # Convertir les timestamps UNIX en datetime

        # Convertir la colonne 'time' en nombre de secondes depuis l'époque pour l'utilisation avec le modèle
        df['time'] = df['time'].astype('int64') / 10**9  # Conversion en secondes depuis l'époque (format float)

        # Calculer la différence de température entre les valeurs successives
        df['temp_diff'] = df['temperature'].diff().abs()

        # Filtrer les données pour ne garder que celles où la différence de température est significative
        df_filtered = df[df['temp_diff'] > 0.1]  # Seuil de 0.1°C pour la variation

        # Normaliser les données de température
        scaler_temp = MinMaxScaler(feature_range=(0, 1))
        df_filtered['temperature_scaled'] = scaler_temp.fit_transform(df_filtered['temperature'].values.reshape(-1, 1))

        # Séparer les données en entrées (X) et sorties (y)
        X = df_filtered['time'].values.reshape(-1, 1)  # Utiliser 'time' comme variable d'entrée
        y = df_filtered['temperature_scaled'].values  # Utiliser 'temperature_scaled' comme variable de sortie normalisée

        # Créer et entraîner le modèle k-NN
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
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


@app.route('/predict_horizon', methods=['GET'])
def predict_horizon():
    try:
        # Horizon de prédiction récupéré depuis les paramètres de la requête (par défaut 10 secondes)
        horizon = int(request.args.get('horizon', 10))

        # Charger les données
        df = pd.read_csv("data/HomeC.csv", usecols=['temperature', 'time'])

        # Préparer les données comme dans les exemples précédents
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df.dropna(subset=['time', 'temperature'], inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
        df['time'] = df['time'].astype('int64') / 10**9

        # Normaliser la température
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['temperature_scaled'] = scaler.fit_transform(df['temperature'].values.reshape(-1, 1))

        # Séparer les entrées et sorties
        X = df['time'].values.reshape(-1, 1)
        y = df['temperature_scaled'].values

        # Entraîner un modèle de régression linéaire
        model = LinearRegression()
        model.fit(X, y)

        # Prédire pour les prochains horizons
        last_time = df['time'].values[-1]
        time_steps = [last_time + (i * 10) for i in range(1, horizon + 1)]  # Prévisions toutes les 10 secondes
        predictions_scaled = model.predict(np.array(time_steps).reshape(-1, 1))

        # Dénormaliser les prédictions
        predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

        # Retourner les prédictions sous forme de JSON
        return jsonify({
            "horizon": horizon,
            "predictions": predictions.flatten().tolist(),
            "unit": "seconds"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



    


if __name__ == '__main__':
    app.run(port=5001)
