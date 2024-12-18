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

if __name__ == '__main__':
    app.run(port=5001)
