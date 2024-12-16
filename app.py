import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Bienvenue sur mon API Flask"})



# Charger le modèle IA
model = load_model("mon_modele.h5")

# URL de l'API InfluxDB
INFLUXDB_API_URL = "http://localhost:5000/data"



@app.route('/predict', methods=['GET'])
def predict():
    # Étape 1 : Récupérer les données depuis l'API InfluxDB
    response = requests.get(INFLUXDB_API_URL)
    if response.status_code != 200:
        return jsonify({"error": "Erreur lors de la récupération des données"}), 500

    # Étape 2 : Préparer les données pour le modèle
    data = response.json()
    temps = [entry["value"] for entry in data]  # Extraire les valeurs de température
    seq_length = 10  # Longueur de la séquence pour la prédiction
    if len(temps) < seq_length:
        return jsonify({"error": "Pas assez de données pour la prédiction"}), 400

    input_data = np.array(temps[-seq_length:]).reshape((1, seq_length, 1))  # Préparation pour LSTM

    # Étape 3 : Faire la prédiction
    prediction = model.predict(input_data).tolist()

    # Étape 4 : Retourner les prédictions
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(port=5001)


if __name__ == '__main__':
    app.run(debug=True)




